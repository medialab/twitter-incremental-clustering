# workshop-event-detection
This repository contains the Python code to reproduce the experiments presented in *Embeddings for event detection and tracking in social media data*.

## Installation

We encourage you to create a virtual environment to install Python 3.8.2. Below are two examples, one with conda, another with pyenv-virtualenv.

### With conda
```bash
conda create -n workshop python=3.8.2
source activate workshop
pip install -U pip setuptools
pip install -r requirements.txt
```

### With pyenv-virtualenv
```bash
pyenv virtualenv 3.8.2 workshop
pyenv activate workshop
pip install -U pip setuptools
pip install -r requirements.txt
```

## Download data
We test our method on 2 datasets, **Event2012** [McMinn et al., 2013] and **Event2018** [Mazoyer et al., 2020]. Follow the instructions by [Cao et al., 2024] [here](https://github.com/SELGroup/HISEvent?tab=readme-ov-file#to-run-hisevent) to download the data. Place the entire ./raw_data folder under the root folder.

## Preprocess data
```bash
python preprocess.py
```
