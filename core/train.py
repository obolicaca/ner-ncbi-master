from pathlib import Path
import joblib

import numpy as np
from torch.utils.data import DataLoader
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


from core.config import config
from core.utils import set_seed
from core.dataset import NCBIDataset
from core.train_helpers import loader_thread_init, model_trainer, model_evaluator
from core.model import NER

def train_ner(dir_pth:str = "",batch_size = None,learning_rate = None,
              drop = None,fold:int = 1,model_path:str = "",*,production:bool=False):
    
    """  Trains the ner model.  """
    # set random states
    if not production: set_seed()

    # get the device ("cpu" or "gpu"), multi-gpu training no supported :(
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get relevant dir info
    dir_pth = Path(dir_pth or config["k-fold"]["FOLD_DATA_PATH"])
    assert dir_pth.exists() and dir_pth.is_dir()
    fold_pth = dir_pth / f"fold{fold}"  # 相当于os.path.join()

    # get number of classes
    n_labels = len(joblib.load(dir_pth.parent / "label_encoder.bin").classes_)

    # prepare datasets 
    train_ds = NCBIDataset(*joblib.load(fold_pth / "train.bin"), loss_plus=True) 
    valid_ds = NCBIDataset(*joblib.load(fold_pth / "val.bin"), loss_plus=True) 
    
    # prepare data loaders 
    bs = batch_size or config["training"]["BATCH_SIZE"]
    train_dl = DataLoader(
        train_ds,
        batch_size = bs, 
        num_workers = config["training"]["NUM_WORKERS"], 
        pin_memory=True, #TODO: not sure if 100% work, need to check some stuff
        worker_init_fn = loader_thread_init if not production else None 
    ) 
    valid_dl = DataLoader(
        valid_ds, 
        # won't need to store gradients, so we can increase the size
        batch_size = bs * 2,
        num_workers = 1 
    )
    
    # create instance of NER model and move to device
    model = NER(n_labels, drop);
    model.to(device)
    
    # prepare optimizer and scheduler
    # we are going to remove weight decay from bias and layer norm as 
    # described in https://huggingface.co/transformers/training.html
    # Hard to provide hard argument why this should be done, but intuitively we can think of
    # say, biases as shift helpers that allow model to activate, say relu etc., so we 
    # let the model figure them our without imposing any limitations
    # Also, for example, bias will not be multiplied by xi, and will be like 1 * grad instead of
    # x1 * grad for last partial derivative, so giving it larger values may not impose such a big problem 
    params = list(model.named_parameters())
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    new_params = [  
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01 #https://github.com/dmis-lab/biobert/blob/master/optimization.py
        },
        {
            "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ] 
    optimizer = AdamW(
        new_params, 
        lr = learning_rate or config["training"]["LR"],
        correct_bias=True
     )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # everybody sets warmup steps to 0, so I guess I'll just do the same 
        num_training_steps=(len(train_ds) / bs) * config["training"]["EPOCHS"]
    ) 
    
                     

    stopping_criteria = config["training"]["EARLY_STOP"] 
    current_miss = 0
    best_loss = np.inf    
    
    for e in range(config["training"]["EPOCHS"]):
        t_loss = model_trainer(model, train_dl, device, optimizer, scheduler)
        v_loss = model_evaluator(model, valid_dl, device)
        print(f"EPOCH: {e +1} |", f"TRAIN LOSS: {t_loss:.4f} |", f"VAL LOSS: {v_loss:.4f}") 
        
        if v_loss < best_loss:
            current_miss = 0
            torch.save(
                model.state_dict(),
                model_path or config["training"]["MODEL_PATH"]
            )
            best_loss = v_loss
        else:
            current_miss += 1
            if current_miss == stopping_criteria:
                print(f"stopping early, val loss didn't improve for {stopping_criteria} epochs")
                break
    
    return best_loss
