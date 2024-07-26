# DIAN
## Dynamic Identity-Guided Attention Network for Visible-Infrared Person Re-identification
![image](https://github.com/gaopenghkbu/DIAN/blob/main/Model_architecture.png)
# DIAN Installation
Create the conda environment (Default installation path), other installation path, use -p to select your own path:
```
conda env create -f environment.yml
```
List all of the environment:
```
conda info -envs
```
To activate the environment:
```
conda activate DIAN
```
# Training
```
python train.py --dataset sysu --gpu 0
```
