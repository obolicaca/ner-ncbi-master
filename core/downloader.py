import gdown
import pathlib
from pathlib import Path
from zipfile import ZipFile
import os
import csv
import joblib

from core.config import config
from sklearn.preprocessing import LabelEncoder

def downloader(pth:str = ""):
    """Helper function which downloads data from gdrive and extracts the contents"""
    data_path = Path(pth or config["downloader"]["RAW_DATA_PATH"])
    data_path.mkdir(parents = True, exist_ok = True)
    
    file_path = str(data_path / "corpus.zip")
    
    gdown.download(config["downloader"]["DATA_URL"], file_path, quiet=True)  
    
    with ZipFile(file_path, "r") as z:
        z.extractall(str(data_path))
    os.remove(file_path)

    _extract_sent_tag_pairs(data_path)


def _extract_sent_tag_pairs(pth:pathlib.Path):
    """
    Extracts tsv files into list of lists which holds
    input and target tag pairs.  Saves the result.
    """ 
    proc_dir = (pth.parent / "proc").mkdir(parents = True, exist_ok = True) 

     
    train_files = ("train_dev.tsv", "devel.tsv") 
    test_files = ("test.tsv",) 
    


    encoder = _make_encoder(pth)

    _extract_files(pth, train_files, encoder, "train")
    _extract_files(pth, test_files, encoder, "test")


def _extract_files(pth:pathlib.Path, files:tuple, encoder, out_file:str):
    x, y = [], [] 
    for _file in files:    
        file_path = pth / _file
        with open(file_path) as f:
            reader = csv.reader(f, delimiter="\t") 

            sample_x, sample_y = [],[] 
            for line in reader: 
                if not line:
                   x.append(sample_x); y.append(sample_y)
                   sample_x, sample_y = [], [] 
                else:
                    sample_x.append(line[0]); sample_y.append(line[1]) 
  
    y = [[a.tolist() for a in encoder.transform(i)] for i in y]

    joblib.dump([x,y], pth.parent / "proc" / (out_file + ".bin")) 

def _make_encoder(pth:pathlib.Path):
    """ We could make this generic but meh """
    le = LabelEncoder()
    le.fit(["O", "B", "I"])
    joblib.dump(le, pth.parent / "label_encoder.bin")
    return le 
