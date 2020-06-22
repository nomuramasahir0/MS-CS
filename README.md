# MSU-HPO
Code for reproducing the results in the paper "**Multi-Source Unsupervised Hyperparameter Optimization**".

If you find this code useful in your research then please cite:

```
@article{nomura2020multi,
  title={Multi-Source Unsupervised Hyperparameter Optimization},
  author={Nomura, Masahiro and Saito, Yuta},
  journal={arXiv preprint arXiv:2006.10600},
  year={2020}
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
