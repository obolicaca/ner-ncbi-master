""" Various utility functions """

import random
import numpy as np
import torch
import os
from sklearn.model_selection import KFold
import joblib
from pathlib import Path

from core.config import config


def set_seed(multi_gpu:bool = False): 
    """ 
    Sets seeds for reproducibility.
    However, recall that even with seeding we will most likely get slighly different results:
    https://pytorch.org/docs/stable/notes/randomness.html
    """

    seed = config["random"]["SEED"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    if multi_gpu: torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True 


def split_into_k_folds(pth:str = "", *, k_folds = None):
    pth = Path(pth or config["k-fold"]["FOLD_DATA_PATH"])
    k_folds = k_folds or config["k-fold"]["N_FOLDS"] 
   
    k_fold_dir = pth.parent / "k-fold"
    k_fold_dir.mkdir(parents=True, exist_ok=True)  

    set_seed()
    
    x, y = joblib.load(pth / "train.bin")    


    skf = KFold(n_splits=k_folds, shuffle=True, random_state = config["random"]["SEED"]) 
    for idx, (t_idx, v_idx) in enumerate(skf.split(x,y)):
        fold_x_path = k_fold_dir / f"fold{idx+ 1}"  
        fold_x_path.mkdir(exist_ok=True)
        
        fold_x_train = []
        fold_y_train = []  
        for i in t_idx:
            fold_x_train.append(x[i])
            fold_y_train.append(y[i]) 
        joblib.dump([fold_x_train, fold_y_train], fold_x_path / "train.bin")
            
        fold_x_val = []
        fold_y_val = []  
        for i in v_idx:
            fold_x_val.append(x[i])
            fold_y_val.append(y[i])  
        joblib.dump([fold_x_val, fold_y_val], fold_x_path / "val.bin")   
