# Boxes_Detection

## Instalation

Setting up an virtual enviroment and installing the requisites

```shell script
python3 -m ensurepip --upgrade
python3 -m venv venv
source /venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run training

```shell script
python3 utils/yolo.py
```

## Run evalutation

```shell script
python3 utils/val.py
```

## Run hyperparameters search

```shell script
python3 utils/par.py
```

## Generate ball_handler.npy file
```shell script
python3 func_ball_handler.py
```

## DatasetTools

download.py, framecatcher.py and framechecker.py compose a suite of scripts to facilitate dataset build. Documentation is contained in the files.
