# SimpleNet

An anomaly detection method based on anomalous feature generation.

## 1. Prerequisites

## 1.1. Environment 

**Python3.8**

**Packages**:
- torch==1.12.1
- torchvision==0.13.1
- numpy==1.22.4
- opencv-python==4.5.1

(Above environment setups are not the minimum requiremetns, other versions might work too.)


## 1.2. Data

Edit `run.sh` to edit dataset class and dataset path.

### 1.2.1. MvTecAD

Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

The dataset folders/files follow its original structure.

## 2. Run

### 2.1. Demo train

Please specicy dataset path (line1) and log folder (line10) in `run.sh` before running.

`run.sh` gives the configuration to train models on MVTecAD dataset.
```
bash run.sh
```