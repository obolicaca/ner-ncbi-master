""" Simple config file that is used to store various info """
from transformers import DistilBertTokenizerFast


# very similar structure to x.ini extension files
config = {
    "downloader" : {
        "RAW_DATA_PATH" : "data/raw",
        "DATA_URL" : "https://drive.google.com/uc?id=1xt7P-zr5U7FB1EIU8rItnqkCgWO-2NWW", 
        "PROC_DATA_PATH" : "data/proc",
        "MODEL_URL" :"https://drive.google.com/uc?id=1umgB586ngx3UyAGAgWMI5FvHpPKn8KJG",
        "ENCODER_URL" :"https://drive.google.com/uc?id=1wODYKYBfMbuBvSozvXvP9t2H89YZQvnG"
    },
    # For NER tasks the recommendation is to use the cased version https://arxiv.org/pdf/1901.08746.pdf
    # We'll be using Fast version which is implemented in Rust. How much faster(if faster at all) I don't know, but I take developers word for it 
    # NOTE: fun fact, I played a bit with cased and uncased distilbertFast tokenizers and it appears that DistilBertTokenizerFast is not working as intended for uncased variant 
    # so take care.
    # I raised an issue just fyi: https://github.com/huggingface/transformers/issues/10650.
    "tokenizer" : {
        "TOKENIZER" : DistilBertTokenizerFast.from_pretrained("distilbert-base-cased", do_lower_case = False),
        "MAX_LEN" : 180 
    }, 
    "random" : {
        "SEED" : 112, 
    },
    "k-fold" : {
        "N_FOLDS": 10,
        "FOLD_DATA_PATH" : "data/folds",
    },
    "model_config" : {
        "PRETRAINED_PATH" : "distilbert-base-cased",
        # Let's just set it to 0.3, if we notice some overfitting, we can up this value for more regularization
        "DROPOUT" : 0.3, 
    },
    "training" : {
        "MODEL_PATH": "data/model.bin",
        # adjust according to your cpu hardware 
        # set to 4 well cuz everybody has at least 4 cores right
        "NUM_WORKERS" : 4,
        # optimal hyperparams recommended by bert authors
        # adiitionaly: https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
        "BATCH_SIZE" : 32,
        # recommended epochs are around 4-5 or so, but your can increase it and early stopp so all good
        "EPOCHS": 5, 
        # kind of what bert authors recommend: https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/
        "LR" : 2e-5, # https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU 
        "EARLY_STOP" : 3 
    }

}
