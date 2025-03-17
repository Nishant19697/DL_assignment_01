# Neural Network Trainer Script

## Overview
This script trains a neural network on the **MNIST** or **Fashion-MNIST** dataset. It supports multiple optimizers, loss functions, weight initialization methods, and integrates with **Weights & Biases (WandB)** for experiment tracking.

## Usage
Run the script using:
```bash
python trainer.py [options]
```
Example:
```bash
python trainer.py -d mnist -e 10 -b 128 -o nadam -lr 0.001 -nhl 5 -sz 128 -a ReLU -log True
```

## Command-Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `-wp`, `--wandb_project` | WandB project name | None |
| `-we`, `--wandb_entity` | WandB entity name | None |
| `-d`, `--dataset` | Dataset choice (`mnist` or `fashion_mnist`) | `mnist` |
| `-e`, `--epochs` | Number of training epochs | `10` |
| `-b`, `--batch_size` | Batch size | `128` |
| `-l`, `--loss` | Loss function (`mean_squared_error` or `cross_entropy`) | `cross_entropy` |
| `-o`, `--optimizer` | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) | `nadam` |
| `-lr`, `--learning_rate` | Learning rate | `0.001` |
| `-m`, `--momentum` | Momentum (for momentum-based optimizers) | `0.5` |
| `-beta`, `--beta` | Beta (for RMSprop) | `0.5` |
| `-beta1`, `--beta1` | Beta1 (for Adam/Nadam) | `0.9` |
| `-beta2`, `--beta2` | Beta2 (for Adam/Nadam) | `0.999` |
| `-eps`, `--epsilon` | Epsilon for numerical stability | `1e-8` |
| `-w_d`, `--weight_decay` | Weight decay (L2 regularization) | `0.0001` |
| `-w_i`, `--weight_init` | Weight initialization method (`random`, `Xavier`, `kaiming`) | `kaiming` |
| `-nhl`, `--num_layers` | Number of hidden layers | `5` |
| `-sz`, `--hidden_size` | Hidden layer size | `128` |
| `-a`, `--activation` | Activation function (`sigmoid`, `tanh`, `ReLU`) | `ReLU` |
| `-log`, `--logging` | Enable WandB logging | `True` |

## Logging
If logging is enabled (`-log True`), the script logs training details to **Weights & Biases (WandB)**. The logged run includes:
- Dataset and training configurations
- Optimizer and loss function details
- Training progress and metrics

## Dependencies
- Python 3
- NumPy
- WandB (`pip install wandb`)

## Execution
The script automatically initializes WandB logging if the project and entity are provided. It then trains the model and logs the results.

### Links 
wandb - https://wandb.ai/nishant19697-indian-institute-of-technology-madras/sweep_trial05/reports/DA6401-Assignment-1--VmlldzoxMTgzMzg1OQ?accessToken=17sg9hqrmo6bdglwzuy68wnxjdkuucottceyn7i3zdiwoln5wqsiijlhgi2ekwbf
github - https://github.com/Nishant19697/DL_assignment_01

## Contact
For any queries or issues, feel free to reach out.
