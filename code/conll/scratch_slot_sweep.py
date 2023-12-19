import argparse

import time
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import numpy as np
from transformers import BertModel, BertTokenizer, TrainingArguments, Trainer, BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
from datasets import load_metric

import wandb

metric_seqeval = load_metric("seqeval")
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class SlotAttention(nn.Module):
    """
    A module that implements the slot attention mechanism
    """
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None, num_slots = None, interpret = False):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            # print("Dots: ", dots.shape)
            # print("Mask: ", mask.shape)

            if mask is not None:
                dots = dots.masked_fill(mask==0, -1e10)

            attn = dots.softmax(dim=1) + self.eps
            
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            attn_return = attn

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        if interpret:
            return slots, attn_return
        else:
            return slots


class PositionwiseFeedforwardLayer(nn.Module):
    """
    A module that implements the positionwise feedforward layers
    """
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x


class EncoderLayer(nn.Module):
    """
    A module that implements a single encoder layer 
    """
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout,
                 num_iters,
                 num_labels, 
                 device,
                 num_layer=1):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        # self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.slot_attention = SlotAttention(num_slots=num_labels, dim=hid_dim, iters=num_iters)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.num_layer = num_layer
        
    def forward(self, src, src_mask=None):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
        # print("In forward enc layer:", src.shape, src_mask.shape)        
        #self attention
        # _src, _ = self.self_attention(src, src, src, src_mask)
        _src = self.slot_attention(src, mask=src_mask)
        
        #dropout, residual connection and layer norm
        if self.num_layer > 1:
            src = self.self_attn_layer_norm(src + self.dropout(_src))
        else:
            src = _src
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class Encoder(nn.Module):
    """
    A module that creates an encoder from a stack of encoderlayers
    """

    def __init__(self, 
                 input_dim, #vocab_size
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout,
                 num_iters,
                 num_labels, 
                 device,
                 max_length = 100,
                 pad_id_token = -100):
        super().__init__()

        self.device = device
        
        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout,
                                                  num_iters,
                                                  num_labels, 
                                                  device,
                                                  num_layer=i+1) 
                                     for i in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.src_pad_idx = pad_id_token

    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1)#.unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
        
    def forward(self, src):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # batch_size = src.shape[0]
        # src_len = src.shape[1]
        
        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # #pos = [batch size, src len]
        
        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src = [batch size, src len, hid dim]
        
        
        for i, layer in enumerate(self.layers):
            if i==0:
                src_mask = self.make_src_mask(src)
                src_mask = src_mask[:, :, :, 0]
                src = layer(src, src_mask)
            else:
                src = layer(src)
    
            
        #src = [batch size, src len, hid dim]
            
        return src

class BERTNer(nn.Module):

    """
    Main model class for NER training
    """

    def __init__(self,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 num_iters,
                 dropout,
                 pad_id_token):
        
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        
        self.enc = Encoder(input_dim=embedding_dim, hid_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads, pf_dim=pf_dim, dropout=dropout, device='cuda:0', max_length = 512, pad_id_token = pad_id_token, num_iters = num_iters, num_labels = output_dim)
        
        # self.out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, input_ids, labels, attention_mask=None):
        
        #text = [batch size, sent len]
                
        # with torch.no_grad():
        embedded = self.bert(input_ids)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        enc_out = self.enc(embedded) #[batch size, src len, hid dim] #[bs, num_slots, hid dim]
        
        hidden = self.dropout(enc_out) #[bs, num_slots, hid dim]
                
        #hidden = [batch size, hid dim]
        
        # output = self.out(hidden) #[batch size, src len, out_dim] if enc_out is kept as 3d tensor #einsum abc,adc->abd
        output = torch.einsum("abc,adc->abd", embedded, hidden) #[batch size, src len, out_dim]
        # return output, labels
        return output, labels
    
def count_parameters(model):
    """
    A function that counts the number of parameters of a given model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(dataset_name="conll2003"):

    """
    Function to load the given dataset in tokenized format
    """

    dataset = load_dataset(dataset_name)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    label_names = dataset["train"].features["ner_tags"].feature.names
    # print(dataset.columns_names)
    #Get the values for input_ids, attention_mask, adjusted labels
    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, padding=True, max_length=100)
        
        total_adjusted_labels = []
        
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []
        
            for word_idx in word_ids_list:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if(word_idx is None):
                    adjusted_label_ids.append(-100)
                elif(word_idx!=prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(int(existing_label_ids[i]))
                    prev_wid = word_idx
                else:
                    label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(int(existing_label_ids[i]))    
            total_adjusted_labels.append(adjusted_label_ids)
        
        #add adjusted labels to the tokenized samples
        tokenized_samples["labels"] = total_adjusted_labels
        # print(len(tokenized_samples["labels"]), len(tokenized_samples["input_ids"]), len(adjusted_label_ids))
        return tokenized_samples

    tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True, remove_columns=['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])

    return tokenized_dataset, label_names

def compute_metrics(p):
    """
    A function that computes metrics for predictions
    """

    predictions, labels = p
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    #select predicted index with maximum logit for each token
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'] 
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric_seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train(model, iterator, optimizer, criterion, device):
    """
    A function that performs traning through iterator of an epoch
    """

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_recall = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        
        predictions, labels = model(batch.input_ids.to(device), batch.labels.to(device))
        
        loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        acc = compute_metrics((predictions, labels))
        accuracy = acc["accuracy"]
        precision = acc["precision"]
        recall = acc["recall"]
        f1 = acc["f1"]
        
        epoch_loss += loss.item()
        epoch_acc += accuracy
        epoch_precision += precision
        epoch_recall += recall
        epoch_f1 += f1

    metrics = dict(accuracy=epoch_acc / len(iterator),
                 precision=epoch_precision / len(iterator),
                 recall=epoch_recall / len(iterator), 
                 f1=epoch_f1 / len(iterator))     
    return epoch_loss / len(iterator), metrics

def evaluate(model, iterator, criterion, device):
    """
    A function to evaluate the model
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_recall = 0
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions, labels = model(batch.input_ids.to(device), batch.labels.to(device))
            
            loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))
            acc = compute_metrics((predictions, labels))
            accuracy = acc["accuracy"]
            precision = acc["precision"]
            recall = acc["recall"]
            f1 = acc["f1"]

            epoch_loss += loss.item()
            epoch_acc += accuracy
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1
    metrics = dict(accuracy=epoch_acc / len(iterator),
                 precision=epoch_precision / len(iterator),
                 recall=epoch_recall / len(iterator), 
                 f1=epoch_f1 / len(iterator))    
    return epoch_loss / len(iterator), metrics

def epoch_time(start_time, end_time):
    """
    A function that computes the time required for an epoch to complete
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main(config=None):

    with wandb.init(config=config):   # Initialisation of wandb

        args = wandb.config

        wandb.run.name = f'{args.out_dim}slots_{args.num_iters}num-iters_{args.out_dim}out_dim_{args.dropout}drop_prob_{args.hid_dim}h_{args.n_layers}l_{args.batch_size}bs_{args.lr}lr_{args.pf_dim}pf_dim'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dataset, label_names = load_data(dataset_name="conll2003")  # Data Loading
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        data_collator = DataCollatorForTokenClassification(tokenizer)

        ## Creaing model
        model = BERTNer(hidden_dim=args.hid_dim, output_dim=args.out_dim, n_layers=args.n_layers, n_heads=args.n_heads,pf_dim=args.pf_dim, dropout=args.dropout, pad_id_token=data_collator.label_pad_token_id, num_iters=args.num_iters).to(device)
        print(f"Model has {count_parameters(model)} number of parameters")

        ## Commented becuase we are finetuning
        # freeze bert layers
        # for name, param in model.named_parameters():                
        #     if name.startswith('bert'):
        #         param.requires_grad = False
        
        print(f"Model has {count_parameters(model)} number of parameters")

        ## Creating data iterators
        train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
        valid_iterator = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=data_collator.label_pad_token_id)

        
        ## Training and validation
        best_valid_loss = float('inf')

        for epoch in range(args.epochs):
            start_time = time.time()
        
            train_loss, train_metrics = train(model, train_iterator, optimizer, criterion, device)
            valid_loss, valid_metrics = evaluate(model, valid_iterator, criterion, device)
                
            end_time = time.time()
                
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model-slot.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_metrics}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_metrics}')


            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_metrics,
                "valid_loss": valid_loss,
                "valid_acc": valid_metrics})

        ## Testing upon completion of training
        test_loss, test_metrics = evaluate(model, test_iterator, criterion, device)
        wandb.log({"test_loss": test_loss, "test_acc": test_metrics, "epoch": args.epochs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hid_dim", type=int, default=768)
    parser.add_argument("--out_dim", type=int, default=7)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--pf_dim", type=int, default=512)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="conll2003")
    args = parser.parse_args()
    sweep_config = {
    'method': 'grid'
    }
    metric = {
    'name': 'valid_loss',
    'goal': 'minimize'   
    }

    sweep_config['metric'] = metric
    parameters_dict = {
    'epochs': {
        'values': [10]
        },
    'lr': {
          'values': [1e-3, 1e-5, 1e-1]
        },
    'batch_size': {
          'values': [32]
        },
    'hid_dim': {
          'values': [768]
    },
    'out_dim': {
          'values': [9] 
    },
    'n_layers': {
          'values': [8]
    },
    'dropout': {
          'values': [0.0, 0.1]
    },
    'n_heads': {
          'values': [7]
    },
    'num_iters': {
          'values': [3]
    },
    'pf_dim': {
          'values': [128, 256, 512]
    },
    'dataset': {
          'value': args.dataset
    }
    }
    
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="sratch-slot-attention-bert-conll2003")
    # main(args)
    wandb.agent(sweep_id, function=main)