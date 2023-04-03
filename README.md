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

## Website tree
```
.
├── apps
│   ├── Analyse.py
│   ├── cbc.py
│   ├── folder.py
│   ├── home.py
│   ├── metrics.py
│   ├── render.py
│   └── wbc.py
├── helper
│   ├── advanced_metrics.py
│   ├── functions.py
│   ├── model.py
│   └── svm.py
├── main.py
├── packages.txt
├── README.md
├── requirements.txt
└── storage
    ├── images
    │   ├── bloodcount
    │   ├── classification
    │   │   ├── BCCD.pkl
    │   │   ├── LISC.pkl
    │   │   └── Raabin.pkl
    │   └── media
    │       ├── dataset.gif
    │       ├── RBC.gif
    │       └── WBC.gif
    ├── models
    │   ├── nn
    │   │   ├── plt.h5
    │   │   ├── rbc.h5
    │   │   └── wbc.h5
    │   └── svm
    │       ├── BCCD.pkl
    │       ├── BCCD_train.npy
    │       ├── LISC.pkl
    │       ├── LISC_train.npy
    │       ├── Raabin.pkl
    │       ├── Raabin_train.npy
    │       └── x_train.npy
    └── tmp
        └── r.md
```
*Why is there a tmp directory?*
> It's for converting streamlits stream objects into np arrays.
> You can also add plt.imsave statements after every predictions or succesful count. This is how the figures of our report were created :)

*How can I contribute?*
> If you'd like to add your own method, you have to put all the logic into the `helper/` directory.
> In `apps/` you'll have to create an entry function and functions for the logic (i.e predictions, rendering output, handling input logic).
> If you'd like to add you module, edit the entry file `main.py` and import your **entry function**.
> To create a module edit the selection bar (The used Component can be found here: https://github.com/victoryhb/streamlit-option-menu 

*I want to use the Cell-U-Net*
> Due to the limitations of the streamlit community cloud a simple model has been used.
> The Cell-U-Net .h5 can be downloaded here https://drive.google.com/file/d/18l18n1g5E5Dl3CxOsrC9BP9rKcsyRH2K/view?usp=sharing

*I want to use my own dataset*
> You can exchange the image file in `storage/BCCD/*` or create your own folder.
> The images are loaded using the `dataset_prediction` function in `helper/svm.py`
> You need to edit the image type and the **ground_truth.json** or delete it, if not present

