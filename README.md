# Website

## Setting up the Enviorment


### Conda
```bash
conda create -n "CellAlyse" python=3.9 -y
conda activate CellAlyse 
pip install -r requirements.txt
```

### venv
**Note that the Website only works with python 3.9**
```bash
python -m venv CellAlyse
source CellAlyse/bin/activate
pip install -r requirements.txt
```

### Running the Website
```bash
streamlit run main.py
```

### Using single-board computers
**Modules related to full blood counts won't be working on less capable hardware**

#### Setting up a suitable conda enviorment
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-Linux-aarch64.sh
bash Miniforge-pypy3-Linux-aarch64.sh
```
**follow the setup process**
*If the enviorment is not working after rebooting run the command with the -u argument*

#### Installing tensorlfow 2.x
```bash
conda create -n "CellAlyse" python=3.7 -y
conda activate CellAlyse
```bash
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install openmpi-bin libopenmpi-dev
sudo apt-get install liblapack-dev
```
*This setup is from https://qengineering.eu/install-tensorflow-2.1.0-on-raspberry-pi-4.html*

```bash
pip install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_aarch64.whl
```
*The wheel is from https://bitsy.ai/3-ways-to-install-tensorflow-on-raspberry-pi/*

```bash
pip install -r requirements-singleboard.txt
```

If you don't won't to open a new window after every run (headless), you can edit the `./streamlit/config.toml` file. Just add:
```toml
[server]

headless = false
```


