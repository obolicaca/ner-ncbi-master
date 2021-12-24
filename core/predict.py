""" predict new inputs and stuff """

import joblib
import torch
import gdown
import numpy as np

from pathlib import Path

from core.config import config 
from core.model import NERPredict

class Predictor:
    
    @staticmethod
    def is_piece(x:str) ->str:
        "if ## is present in original str it will get pieced so all good"
        return x[:2] == "##"
    @staticmethod
    def is_full(x:str) ->str: return x[:2] != "##" 
    
    def _get_encoder(self):
        enc_pth = self.pth / "label_encoder.bin"
        if not enc_pth.exists():
            gdown.download(config["downloader"]["ENCODER_URL"], str(enc_pth), quiet=True)
        return joblib.load(enc_pth)
            
    def _prepare_model(self): 
        model_pth = self.pth / "model.bin"
        if not model_pth.exists():
            gdown.download(config["downloader"]["MODEL_URL"], str(model_pth), quiet=True)
            
        model = NERPredict(len(self.enc.classes_))
        model.load_state_dict(torch.load(model_pth))
        model.eval()
        return model    

    def _align_preds(self, texts, y_hat):
        # well, this is technically ammortizied O(n)  
        # but I will deny that I made this mostrocity is asked
        out_txt = []
        out = []
        idx = 0
        pooled = []
        pooled_txt = []
        while idx < (len(texts) -1):
            
            if self.is_full(texts[idx]) and self.is_full(texts[idx + 1]):
                out.append(np.argmax(y_hat[idx])) 
                out_txt.append(texts[idx]) 
                idx += 1  
            elif self.is_full(texts[idx]) and self.is_piece(texts[idx+1]):
                pooled.append(y_hat[idx])
                pooled_txt.append(texts[idx]) 
                idx += 1 
                while self.is_piece(texts[idx]): 
                    pooled.append(y_hat[idx])
                    pooled_txt.append(texts[idx][2:])  
                    idx += 1 
                out.append(np.argmax(np.mean(np.array(pooled), axis=0)))
                out_txt.append("".join(pooled_txt)) 
                pooled = [] 
                pooled_txt = []
            elif self.is_piece(texts[idx]):
                pooled.append(y_hat[idx]) 
                pooled_txt.append(texts[idx][2:]) 
                idx += 1  
                while self.is_piece(texts[idx]):
                    pooled.append(y_hat[idx])
                    pooled_txt.append(texts[idx][2:])  
                    idx += 1 
                pooled.append(y_hat[idx]) 
                pooled_txt.append(texts[idx])  
                idx += 1
                while self.is_piece(texts[idx]):
                    pooled.append(y_hat[idx])
                    pooled_txt.append(texts[idx][2:])  
                    idx += 1
                out.append(np.argmax(np.mean(np.array(pooled), axis=0)))
                out_txt.append("".join(pooled_txt)) 
                pooled = []        
                pooled_txt = []
        return out_txt, self.enc.inverse_transform(out).tolist() 
    
    def __init__(self, pth:str):
        self.pth = Path(pth)
        self.tok = config["tokenizer"]["TOKENIZER"]
        self.max_len = config["tokenizer"]["MAX_LEN"]
        self.enc = self._get_encoder() 
        self.model = self._prepare_model()
    
    def _predict_one(self, text:str):
        tokenized = self.tok.tokenize(text)
        x = [101] + self.tok.encode(text, add_special_tokens=False)[:self.max_len - 2] + [102]
        mask = [1 for _ in range(len(x))]  
        
        x = x + [0 for _ in range(self.max_len - len(x))]
        mask = mask + [0 for _ in range(self.max_len - len(mask))]
        
        x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            out = self.model({"input_ids" : x, "attention_mask" : mask}).numpy()
       
        out = out[:, 1:len(tokenized) + 2][0]
        texts , predictions = self._align_preds(tokenized + ["XXX"], out)
        return texts, predictions

    def __call__(self, text:str):
        return self._predict_one(text)
