## Effective Function-Space Regularization for Bayesian Neural Networks via Learning Trajectories

This is an implemenation for "Effective Function-Space Regularization for Bayesian Neural Networks via Learning Trajectories"



## Description for main implementation

- `models_deepnet.py`: MAP and proposed ResNet models.
- `models_vit.py`: ViT models for transfer learning.
- `train_bmafbnn.py`: Train R-FVI with last-layer stochastic ResNets.
- `train_bmafbnn_st.py`: Train R-FVI with fully stochastic ResNets.
- `train_vit_bmafbnn.py`: Train R-FVI with last-layer stochastic ViTs.

## Example to run code

Set the project directory and GPU. Python bytecode files are disabled.

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
```

Train fully stochastic ResNet18 on CIFAR-10

```bash
bash run_resnet18st_seed1111.sh
```

Train fully stochastic ResNet50 on CIFAR-100

```bash
bash run_resnet50st_seed1111.sh
```

Train ViT MAP on DTD 

```bash
bash run_vit_map_seed1111.sh
```

Train the proposed ViT method on DTD

```bash
bash run_vit_proposed_seed1111.sh
```
