#! /bin/bash

conda create --name dancedancetransformation python=3.7
conda activate dancedancetransformation
conda install cudatoolkit=10.0

pip install -r requirements.txt
