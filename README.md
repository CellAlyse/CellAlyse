# Website

## Setting up the Enviorment


## Conda
```bash
conda create -n "CellAlyse" python=3.9 -y
conda activate CellAlyse 
pip install -r requirements.txt
```

## venv
**Note that the Website only works with python 3.9**
```bash
python -m venv CellAlyse
source CellAlyse/bin/activate
pip install -r requirements.txt
```

## Running the Website
```bash
streamlit run main.py
```

### Using single-board computers
**Modules related to full blood counts won't be working on less capable hardware**

#### Setting up a suitable conda enviorment
``wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-Linux-aarch64.sh``
follow the setup process
** 
