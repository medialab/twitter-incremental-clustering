# Workshop 2024 on Event Detection
This repository contains the Python code to reproduce the experiments presented in our paper:

*An Incremental Clustering Baseline for Event Detection on Twitter*.

## Table of contents
- [Installation](#installation)
- [Download data](#download-data)
- [Preprocess data](#preprocess-data)
- [Run event detection](#run-event-detection)
- [Generate latex table](#generate-latex-table)
- [Plot execution time](#plot-execution-time)

## Installation

We encourage you to create a virtual environment to install Python 3.8.2. Below are two examples, one with conda, another with pyenv-virtualenv.

### With conda
```bash
git clone https://github.com/medialab/twitter-incremental-clustering.git
cd twitter-incremental-clustering
conda create -n workshop python=3.8.2
source activate workshop
pip install -U pip setuptools
pip install -r requirements.txt
```

### With pyenv-virtualenv
```bash
git clone https://github.com/medialab/twitter-incremental-clustering.git
cd twitter-incremental-clustering
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

## Run event detection
1. Run event detection on **Event2018** dataset with
   **[Sentence-CamemBERT Large](https://huggingface.co/dangvantuan/sentence-camembert-large)** (GPU required):
    ```bash
    python run_detection.py --model sbert --sub-model "dangvantuan/sentence-camembert-large" --lang fr --dataset event2018.tsv
    ```
2. Run event detection on **Event2012** dataset with **[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)** (GPU required):
   ```bash
    python run_detection.py --model sbert --sub-model "sentence-transformers/all-mpnet-base-v2" --lang en --dataset event2012.tsv
    ```

## Generate latex table
```bash
python generate_table.py
```
The table is saved in `ami_ari_metrics.tex`

## Plot execution time
After running the event detection several times with several --batch-size values, plot the effect of the parameter on AMI and execution time with the command:
```bash
python plot_time.py
```
The figure is saved in `timeplot.pdf`
