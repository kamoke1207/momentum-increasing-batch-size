# Additional Experiments for Rebuttal
This document summarizes the additional experiments we conducted in response to the reviewer's questions regarding the practical advantages of increasing batch size strategies, particularly under realistic constraints such as GPU memory limitations and fixed computational budgets.

## 1. Definition: Stochastic First-order Oracle (SFO) Complexity
Let $b$ be a batch size.
The deep neural network model uses $b$ gradients of the loss function per step.
Hence, when $T$ is the number of steps required to train a deep neural network, the model has a stochastic gradient computation cost of $Tb$.

The quantity $Tb$ represents the total number of stochastic gradient calls, which we refer to as the Stochastic First-order Oracle (SFO) complexity.

## 2. Experimental Motivation
To address the reviewer’s concerns regarding the practicality of increasing batch size strategies under GPU memory constraints and the fairness of comparing different schedules under equal epoch counts, we conducted additional experiments focusing on gradient norm thresholds and test accuracy, evaluated under a fixed computational budget measured by stochastic gradient computations (SFO).

### 2.1 Focus of this Study
In response to your questions regarding whether increasing batch size still offers advantages under a fixed computational budget—specifically in terms of empirical loss and test accuracy—we conducted additional experiments that compare fixed and increasing batch size strategies based on the SFO complexity required to reach specific gradient norm and accuracy thresholds.

While we initially aimed to compare fixed and increasing batch size schedules under the same total gradient computation budget, we now instead focus on comparing the SFO complexity required for each method to reach a specified value of the full gradient norm during training. We believe this offers a more meaningful and performance-driven evaluation.

### 2.2 Experimental Design Choices
We conducted the experiments using the CIFAR-100 dataset and focused on the NSHB optimizer only, while keeping all other training settings consistent with those described in the main paper. This ensures that performance differences arise solely from batch size scheduling, not from confounding factors such as architecture or hyperparameters. Specifically, we conducted experiments under the following settings:

## 3. Experimental Settings
### 3.1 For Gradient Norm Threshold Evaluation 
  - Fixed: batch size b = 8, Increasing: initial batch size = 8, doubling every 20 epochs
  - Fixed: batch size b = 128, Increasing: initial batch size = 128, doubling every 50 epochs

### 3.2 For Test Accuracy Threshold Evaluation
  - Fixed: batch size b = 8, Increasing: initial batch size = 8, doubling every 20 epochs
  - Fixed: batch size b = 128, Increasing: initial batch size = 128, doubling every 25 epochs

To reflect practical hardware limitations, we capped the maximum batch size at 1024 to simulate realistic GPU memory constraints. This ensures that our results remain applicable to real-world training environments.

The results of these experiments are summarized in the two tables below.

## 4. Results
### 4.1 Gradient Norm Thresholds

| Method | SFO to reach $\|\nabla f\|$ < 0.1 | SFO to reach $\|\nabla f\|$ < 0.05 |
|---------|---------|---------|
| Fixed batch size (b=8) | 2,750,000 | 5,250,000 |
| Increasing batch size (initial b=8) | 2,050,016 | 2,500,160 |

| Method | SFO to reach $\|\nabla f\|$ < 0.1 | SFO to reach $\|\nabla f\|$ < 0.06 |
|---------|---------|---------|
| Fixed batch size (b=128) | 5,755,520 | 9,809,408 |
| Increasing batch size (initial b=128) | 2,903,808 | 5,061,376 |


### 4.2 Test Accuracy Threshold (70%)

| Method | SFO to reach 70% test accuracy |
|---------|---------|
| Fixed batch size (b=8) | 5,250,000 |
| Increasing batch size (initial b=8) | 2,050,016 |

| Method | SFO to reach 70% test accuracy |
|---------|---------|
| Fixed batch size (b=128) | 2,502,400 |
| Increasing batch size (initial b=128) | 1,301,376 |

## 5. Conclusion
These results demonstrate that increasing batch size reaches the same gradient norm threshold using significantly fewer stochastic gradient computations, particularly for lower thresholds. This supports the practical and theoretical relevance of increasing batch size strategies, particularly in realistic nonconvex optimization settings where constant batch size assumptions are often not feasible.

These findings and discussions will be incorporated into the revised manuscript.
