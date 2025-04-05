This folder contains additional experimental results for the rebuttal.

Fixed: batch size b = 8, Increasing: initial batch size = 8, doubling every 20 epochs
Fixed: batch size b = 128, Increasing: initial batch size = 128, doubling every 50 epochs

The results of these experiments are summarized in the two tables below.

**Gradient Norm Thresholds**

| Method | SFO to reach $\|\nabla f\| < 0.1$ | SFO to reach $\|\nabla f\| < 0.05$ |
|---------|---------|---------|
| Fixed batch size (b=8) | 2,750,000 | 5,250,000 |
| Increasing batch size (initial b=8) | 2,050,016 | 2,500,160 |

| Method | SFO to reach $\|\nabla f\| < 0.1$ | SFO to reach $\|\nabla f\| < 0.06$ |
|---------|---------|---------|
| Fixed batch size (b=128) | 5,755,520 | 9,809,408 |
| Increasing batch size (initial b=128) | 2,903,808 | 5,061,376 |


Fixed: batch size b = 8, Increasing: initial batch size = 8, doubling every 20 epochs
Fixed: batch size b = 128, Increasing: initial batch size = 128, doubling every 25 epochs

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
