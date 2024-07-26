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
python train.py --dataset sysu --gpu 0 --img_w 144 --img_h 288 --batch-size 6 --num_pos 4 --trial --lr 0.1 --erasing_p 0.5 
```
```
--dataset choose the dataset for training (SYSU-MM01, RegDB)
--gpu select the GPU device
```
For dataset acquisition, subscribe via https://github.com/wuancong/SYSU-MM01, http://dm.dongguk.edu/link.html
