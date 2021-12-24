""" Various helper functions for training """

import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

from core.config import config

def loader_thread_init(thread_id):
    # https://www.gitmemory.com/issue/pytorch/pytorch/7068/484918113
    # if we want to ensure repoducible results,
    # we should also seed each thread in data loader if we using
    # multiple hard threads (I thinks)
    np.random.seed(np.random.seed(config["random"]["SEED"]))

def model_trainer(model, dl, device, opt, scheduler):
    """ Fn that trains the model for 1 epoch """ 
    model.train()

    e_loss =  0 
    for batch in tqdm(dl, total = len(dl)):
        opt.zero_grad()

        for k, v in batch.items(): batch[k] = v.to(device) 
        _, loss = model(batch)

        # I'm not 100 % sure if AdamW does grad clipping, should probably check it out
        # But just in case let's clip here
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        opt.step()
        scheduler.step() 
        e_loss += loss.item()

    return e_loss / len(dl)

def model_evaluator(model, dl, device):
    """ Fn that evaluated the model on validation set """
    model.eval() 

    e_loss = 0
    with torch.no_grad():
        for batch in tqdm(dl, total = len(dl)): 
            for k, v in batch.items(): batch[k] = v.to(device) 
            _, loss = model(batch) 
            e_loss += loss.item()

    return  e_loss / len(dl) 
