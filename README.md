# MSU-HPO
Code for reproducing the results in the paper "Multi-Source Unsupervised Hyperparameter Optimization".

## Abstract
__*How can we conduct efficient hyperparameter optimization for a completely new task?*__
In this work, we consider a novel setting, where we search for the optimal hyperparameters for a target task of interest using only unlabeled target task and ‘somewhat relevant’ source task datasets.
In this setting, it is essential to estimate the ground-truth target task objective using only the available information.
We propose estimators to unbiasedly approximate the ground-truth with a desirable variance property.
Building on these estimators, we provide a general and tractable hyperparameter optimization procedure for our setting.
The experimental evaluations demonstrate that the proposed framework broadens the applications of automated hyperparameter optimization.

## Environments

* Python 3.7.1
* numpy==1.16.3
* scipy==1.3.0
* lightgbm==2.2.3
* densratio==0.2.2
* scikit-learn==0.21.2

## Command to Reproduce

### Toy Problem

```bash
./bash/mssynthetic_run.sh
```

### Parkinson (SVM)

```bash
./bash/parkinson_run.sh
```

### GvHD (LightGBM)

```bash
./bash/gvhd_run.sh
```
