# Increasing Batch Size Improves Convergence of Stochastic Gradient Descent with Momentum

## Abstract
Stochastic gradient descent with momentum (SGDM), which is defined by adding a momentum term to SGD, has been well studied in both theory and practice.
Theoretically investigated results showed that the settings of the learning rate and momentum weight affect the convergence of SGDM. 
Meanwhile, practical results showed that the setting of batch size strongly depends on the performance of SGDM. 
In this paper, we focus on mini-batch SGDM with constant learning rate and constant momentum weight, which is frequently used to train deep neural networks in practice. 
The contribution of this paper is showing theoretically that using a constant batch size does not always minimize the expectation of the full gradient norm of the empirical loss in training a deep neural network, whereas using an increasing batch size definitely minimizes it, that is, increasing batch size improves convergence of mini-batch SGDM.  
We also provide numerical results supporting our analyses, indicating specifically that mini-batch SGDM with an increasing batch size converges to stationary points faster than with a constant batch size. 
Python implementations of the optimizers used in the numerical experiments are available at {...}.
