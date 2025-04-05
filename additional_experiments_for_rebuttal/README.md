To address the reviewer’s concerns regarding the practicality of increasing batch size strategies under GPU memory constraints and the fairness of comparing different schedules under equal epoch counts, we conducted additional experiments focusing on gradient norm thresholds and test accuracy, evaluated under a fixed computational budget measured by stochastic gradient computations (SFO).
In response to your suggestion that the paper would benefit from a more thorough evaluation of varying batch size strategies, we carefully designed experiments to provide a more realistic and performance-driven comparison between fixed and increasing batch size schedules.

While we initially aimed to compare fixed and increasing batch size schedules under the same total gradient computation budget, we now instead focus on comparing the SFO complexity required for each method to reach a specified value of the full gradient norm during training. We believe this provides a more meaningful and performance-oriented evaluation.

We conducted the experiments using the CIFAR-100 dataset and focused on the NSHB optimizer only, while keeping all other training settings consistent with those described in the main paper. Specifically, we conducted experiments under the following settings:

- Fixed: batch size b = 8, Increasing: initial batch size = 8, doubling every 20 epochs
- Fixed: batch size b = 128, Increasing: initial batch size = 128, doubling every 50 epochs

The results of these experiments are summarized in the two tables below.

**Gradient Norm Thresholds**

| Method | SFO to reach $\|\nabla f\|$ < 0.1 | SFO to reach $\|\nabla f\|$ < 0.05 |
|---------|---------|---------|
| Fixed batch size (b=8) | 2,750,000 | 5,250,000 |
| Increasing batch size (initial b=8) | 2,050,016 | 2,500,160 |

| Method | SFO to reach $\|\nabla f\|$ < 0.1 | SFO to reach $\|\nabla f\|$ < 0.06 |
|---------|---------|---------|
| Fixed batch size (b=128) | 5,755,520 | 9,809,408 |
| Increasing batch size (initial b=128) | 2,903,808 | 5,061,376 |


- Fixed: batch size b = 8, Increasing: initial batch size = 8, doubling every 20 epochs
- Fixed: batch size b = 128, Increasing: initial batch size = 128, doubling every 25 epochs

The results of these experiments are summarized in the two tables below.

**Test Accuracy Threshold**

| Method | SFO to reach 70% test accuracy |
|---------|---------|
| Fixed batch size (b=8) | 5,250,000 |
| Increasing batch size (initial b=8) | 2,050,016 |

| Method | SFO to reach 70% test accuracy |
|---------|---------|
| Fixed batch size (b=128) | 2,502,400 |
| Increasing batch size (initial b=128) | 1,301,376 |

These results demonstrate that increasing batch size reaches the same gradient norm threshold using significantly fewer stochastic gradient computations, particularly for lower thresholds. This supports the practical and theoretical relevance of increasing batch size strategies, particularly in realistic nonconvex optimization settings where constant batch size assumptions are often not feasible.

We promise to add the above discussion to the main body in the revised manuscript.
