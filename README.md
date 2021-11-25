# MS-CS
Code for reproducing the results in the paper "**Efficient Hyperparameter Optimization under Multi-Source Covariate Shift**".

If you find this code useful in your research then please cite:

```
@inproceedings{nomura2021efficient,
  title={Efficient Hyperparameter Optimization under Multi-Source Covariate Shift},
  author={Nomura, Masahiro and Saito, Yuta},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={1376--1385},
  year={2021}
}
```

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
