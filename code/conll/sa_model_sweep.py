import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class MultiHeadAttentionLayer(nn.Module):
    """
    A module that implements the multi-head attention mechanism
    """
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        # print(query.shape, key.shape, value.shape)        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class EncoderLayer(nn.Module):
    """
    A module that implements a single encoder layer 
    """
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
        # print("In forward enc layer:", src.shape, src_mask.shape)        
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
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
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
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
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.src_pad_idx = pad_id_token

    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
        
    def forward(self, src):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        src_mask = self.make_src_mask(src)
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        src_mask = src_mask[:, :, :, :, 0]
        # print("src shape: ",src.shape)
        # print("src_mask shape: ",src_mask.shape)
        for layer in self.layers:
            src = layer(src, src_mask)
            
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
                 dropout,
                 pad_id_token):
        
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        
        self.enc = Encoder(input_dim=embedding_dim, hid_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads, pf_dim=pf_dim, dropout=dropout, device='cuda:0', max_length = 512, pad_id_token = pad_id_token)
        
        self.out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, input_ids, labels, attention_mask=None):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(input_ids)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        enc_out = self.enc(embedded) #[batch size, src len, hid dim]
        
        hidden = self.dropout(enc_out)
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden) #[batch size, src len, out_dim] if enc_out is kept as 3d tensor
        
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
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, truncation=True, max_length=100)
        
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
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = word_idx
                else:
                    label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i])    
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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

    with wandb.init(config=config):  # Initialisation of wandb

        args = wandb.config

        wandb.run.name = f'{args.n_heads}heads_{args.out_dim}out_dim_{args.dropout}drop_prob_{args.hid_dim}h_{args.n_layers}l_{args.batch_size}bs_{args.lr}lr_{args.pf_dim}pf_dim'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dataset, label_names = load_data(dataset_name="conll2003") # Data Loading
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        data_collator = DataCollatorForTokenClassification(tokenizer)

        ## Creaing model
        model = BERTNer(hidden_dim=args.hid_dim, output_dim=args.out_dim, n_layers=args.n_layers, n_heads=args.n_heads,pf_dim=args.pf_dim, dropout=args.dropout, pad_id_token=data_collator.label_pad_token_id).to(device)
        print(f"Model has {count_parameters(model)} number of parameters")

        # freeze bert layers
        for name, param in model.named_parameters():                
            if name.startswith('bert'):
                param.requires_grad = False
        
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
                torch.save(model.state_dict(), 'model.pt')
            
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
        'values': [5, 10]
        },
    'lr': {
          'values': [1e-3, 1e-2]
        },
    'batch_size': {
          'values': [64, 128]
        },
    'hid_dim': {
          'values': [768]
    },
    'out_dim': {
          'values': [9]
    },
    'n_layers': {
          'values': [2, 4]
    },
    'dropout': {
          'values': [0.0, 0.1]
    },
    'n_heads': {
          'values': [2, 4, 8, 16]
    },
    'pf_dim': {
          'values': [256, 512]
    },
    'dataset': {
          'value': args.dataset
    }
    }
    
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="ner-sweep-self-attention-conll2003")
    # main(args)
    wandb.agent(sweep_id, function=main)