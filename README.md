# NER for NCBI Disease
This repo contains code for training Named Entity Recognition ([NER](https://en.wikipedia.org/wiki/Named-entity_recognition)) task for [NCBI disease](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/) dataset, by fine-tuning [distilbert](https://arxiv.org/pdf/1910.01108.pdf).

# Installation
You can use your prefered  manager (conda pyenv, pipenv etc etc.) and install from requirements.txt
* NOTE: requirements.txt does not include torch, as I'm using pipenv and it breaks with cuda+11, so I have to manually install it using:
* "pipenv run pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html"
* so please install torch according to your specs: https://pytorch.org/get-started/locally/ 

# HOW TO
The repo is meant to be read via jupyter notebooks inside notebooks directory.
If you'd like to just play around with predictions given an arbitrary string, you an either:

* run notebook "notebooks/Playground.ipynb"
* run fastapi API via "python app.py"

### Feel free to contact me if you're encountering issues with regard to running the code
