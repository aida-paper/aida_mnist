# AIDA MNIST

This package contains code related to the Action Inquiry DAgger (AIDA) paper.
In particular, the code is this package can be used to recreate results related to the Sensitivity-Aware Gating (SAG) algorithm on the MNIST dataset.
The SAG algorithm allows one in an interactive learning setting to maintain a desired sensitivity throughout training.
This means that if one sets the desired sensitivity is set to 0.9, SAG will ensure that the novice queries the teacher in 90% of the cases that the novice's prediction is wrong.
This allows to easily trade-off the number of queries vs the number of novice failures by maintaining a prespecified sensitivity level during training.

## Installation Instructions

### Prerequisites: install `uv`

It is adviced to use uv to install the dependencies of `aida_mnist` package.
Please make sure `uv` is installed according to the [installation instructions](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

### Install `aida_mnist`

First clone and go to the `aida_mnist` folder:

```bash
git clone https://github.com/aida-paper/aida_mnist.git
cd aida_mnist
```

Create a virtual environment:

```bash
uv venv
```

Source the virtual environment:

```bash
source .venv/bin/activate
```

Install the `aida_mnist` package`:

```
uv pip install -e .
```

## Main training script

The main training script can be run as follows.
In case you have a CUDA-enabled GPU you can run:

```bash
python ./scripts/main.py --reps 2 --s_des 0.9
```

Otherwise, for CPU training run:

```bash
python ./scripts/main.py --reps 2 --s_des 0.9 --accelerator cpu
```

To reproduce the experiments from the paper run:

```bash
python ./scripts/main.py
```

To also reproduce the ablations from the paper run:

```bash
python ./scripts/ablations.py
```

#### Interactive training

This will train LeNet model(s) interactively with SAG on the MNIST dataset.
The training procedure goes as follows.
Every time step, `[batch_size]` novel images of digits are sampled from the MNIST dataset.
Then we perform inference with the LeNet model (the novice) on these images and quantify the model's uncertainty for each sample.
Using SAG, the theshold is determined for gating.
For every sample with an uncertainty level that exceeds this threshold, a ground truth label is queried.
Also, for the samples with an uncertainty lower than the threshold, a ground truth is label is queried with a probability of `[p_rand]`.
All samples for which a ground truth label is queried are added to the training dataset.
Finally, the model is updated with the training dataset every `[update_every]` steps.

#### Uncertainty quantification
Uncertainty quantification is performed through Monte-Carlo dropout with a dropout rate of 40% and 16 dropout evaluations.
This means there is an ensemble $\mathcal{C} = h_1, \dots, h_C$.
For samples $x$ with labels $y$, the uncertainty is defined as

$u = 1 - \max_y P_\mathcal{C}(y|x)$,

where $P_\mathcal{C}(y|x) = \frac{1}{C} \sum_{i=0}^C P_i(y|x)$.


## Download results

Instead of training the models yourself, it is also possible to download the results data from the experiments in the paper.

```bash
python scripts/download_results.py
```

## Reproduce plots in paper

After training or downloading the results, you can plot the results as in the paper by doing:

```bash
python ./scripts/plot_results.py
```

The resulting figure is save at `figures/mnist.pdf`.

After downloading the results or performing the ablation experiments, you can plot the ablations plots by:

```bash
python ./scripts/plot_reg_albation.py
```

and

```bash
python ./scripts/plot_prand_albation.py
```

## Acknowledgements

This work uses code from the TorchUncertainty open-source project.

#### TorchUncertainty
Original:  [https://github.com/torch-uncertainty/torch-uncertainty](https://github.com/torch-uncertainty/torch-uncertainty)  
License: [Apache 2.0](https://github.com/torch-uncertainty/torch-uncertainty/blob/main/LICENSE)    
Changes: Our main training script is adapted from this [tutorial](https://torch-uncertainty.github.io/auto_tutorials/tutorial_mc_dropout.html#sphx-glr-auto-tutorials-tutorial-mc-dropout-py).
The data modules are modified to allow for interactive training with a subset of the MNIST dataset.