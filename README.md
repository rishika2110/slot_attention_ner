# Utilizing Slot Attention for NER

This repo contains the code for the project "On utilizing Slot Attention for Named Entity Recognition".

Files and their working:
1. For wikiann dataset
`code/wikiann/sa_model_sweep.py` : Runs a wandb sweep for setting utilizing multi-head attention mechanism while keeping the embeddings frozen.
`code/wikiann/slot_model_sweep.py` : Runs a wandb sweep for setting utilizing slot attention mechanism while keeping the embeddings frozen.

`code/wikiann/scratch_self_Sweep.py` : Runs a wandb sweep for setting utilizing multi-head attention mechanism while finetuning the embeddings.
`code/wikiann/scratch_slot_sweep.py` : Runs a wandb sweep for setting utilizing slot attention mechanism while finetuning the embeddings.

2. For CONLL2003 dataset:


Wandb project reports:
1. https://wandb.ai/rishika2110/ner-sweep-slot-attention-mask-wikiann/reports/Slot-Attention-Wikiann--Vmlldzo2MzA2NzYx
2. https://wandb.ai/rishika2110/ner-sweep-self-attention/reports/Multi-head-Attention-Wikiann--Vmlldzo2MzA2ODI1
3. https://wandb.ai/rishika2110/sratch-self-attention-wikiann-init/reports/Multi-head-Attention-Wikiann--Vmlldzo2MzA2Nzk5
4. https://wandb.ai/rishika2110/sratch-slot-attention-bert-wikiann/reports/Slot-attention-Wikiann--Vmlldzo2MzA2ODEx
5. https://wandb.ai/rishika2110/ner-sweep-self-attention-conll2003/reports/Multi-head-attention-CONLL2003--Vmlldzo2MzA2Nzg2
6. https://wandb.ai/rishika2110/ner-sweep-slot-attention-conll/reports/Slot-attention-CONLL2003--Vmlldzo2MzA2ODM5

