""" Implementation of active loss based on:
https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html#DistilBertForTokenClassification """
import torch
from torch.nn import CrossEntropyLoss 

def loss_fn(out:torch.tensor,
            targets:torch.tensor,
            attention_mask:torch.tensor,
            n_labels:int):
    
    """
    Modified cross-entropy loss.
    Only looks at non-masked outputs.
    Additionally, if NSBIDataset initialized with "loss_plus=True",
    it also doesn't calculate loss w.r.t [CLS] and [SEP] tokens. 
    """
    _loss_fn = CrossEntropyLoss()
     
    active_loss = attention_mask.view(-1) == 1     # active_loss为bool类型
    active_logits = out.view(-1, n_labels) 
    
    active_targets = torch.where(                  # 如果active_loss为True,则返回targets.view(-1),否则返回torch.tensor(_loss_fn.ignore_index).type_as(targets)
        active_loss,
        targets.view(-1),
        torch.tensor(_loss_fn.ignore_index).type_as(targets)
    )   
    return _loss_fn(active_logits, active_targets) 
