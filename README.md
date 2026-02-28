# CS336 Spring 2025 Assignment 1: Basics

- You can reproduce the results by running the scripts provided in the ./scripts directory.
- The current implementation supports the TinyStories dataset. If you wish to use OpenWebText, please update the configuration and training scripts accordingly.

## Results

<p align="center">

| Split | TinyStories |
| :--- | ---: |
| **Train Loss** | 1.445 |
| **Valid Loss** | 1.444 |

</p>

> Train loss was measured as the EMA of the final steps with $\beta=0.9$, while validation loss was calculated as the overall average.

## Ablation Results
<p align="center">
  <img src="results/train_result.png" width="600" /><br>
  <img src="results/lr_sweep.png" width="600" /><br>
  <img src="results/batch_sweep.png" width="600" /><br>
  <img src="results/lr_sweep_wandb.png" width="600" /><br>
  <img src="results/batch_sweep_wandb.png" width="600" />
</p>