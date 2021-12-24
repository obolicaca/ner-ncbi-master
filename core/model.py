""" NER model for tagging """

from torch import nn
import transformers

from core.config import config
from core.active_loss import loss_fn

class NER(nn.Module):    

    def __init__(self, n_labels:int, drop = None) -> None:
        super().__init__()
        self.n_labels = n_labels 
        self.dbert = transformers.DistilBertModel.from_pretrained(config["model_config"]["PRETRAINED_PATH"], return_dict=False)
        self.drop = nn.Dropout(drop or config["model_config"]["DROPOUT"])
        #TODO: check initialization and replace if needed
        self.logits = nn.Linear(self.dbert.config.dim, self.n_labels)

    
    def forward(self, x:dict):
        out = self.dbert(input_ids = x["input_ids"], attention_mask = x["attention_mask"])[0]       # DistilBertModel()[0] 的输出是最后一层的last_hidden_state
        out = self.drop(out)
        out = self.logits(out) 
        return out, loss_fn(out, x["targets"], x["attention_mask"], self.n_labels)

class NERPredict(NER):
    def __init__(self, n_labels:int):
        super().__init__(n_labels)
    def forward(self, x:dict):
        out = self.dbert(input_ids = x["input_ids"], attention_mask = x["attention_mask"])[0] 
        return self.logits(self.drop(out))

if __name__ == "__main__":
    model = NER(11)
    print(model)
