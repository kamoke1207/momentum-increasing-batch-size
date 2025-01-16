# Increasing Batch Size Improves Convergence of Stochastic Gradient Descent with Momentum

## Abstract
Stochastic gradient descent with momentum (SGDM), which is defined by adding a momentum term to SGD, has been well studied in both theory and practice.
Theoretically investigated results showed that the settings of the learning rate and momentum weight affect the convergence of SGDM. 
Meanwhile, practical results showed that the setting of batch size strongly depends on the performance of SGDM. 
In this paper, we focus on mini-batch SGDM with constant learning rate and constant momentum weight, which is frequently used to train deep neural networks in practice. 
The contribution of this paper is showing theoretically that using a constant batch size does not always minimize the expectation of the full gradient norm of the empirical loss in training a deep neural network, whereas using an increasing batch size definitely minimizes it, that is, increasing batch size improves convergence of mini-batch SGDM.  
We also provide numerical results supporting our analyses, indicating specifically that mini-batch SGDM with an increasing batch size converges to stationary points faster than with a constant batch size. 
Python implementations of the optimizers used in the numerical experiments are available at https://anonymous.4open.science/r/momentum-increasing-batch-size-888C/.

# Usage
To train a model on **CIFAR-100**, run `cifar100.py` with a JSON file specifying the training parameters. 

```bash
python cifar100.py XXXXX.json
```

To resume training from a checkpoint, add the `--resume` option to the command. This will load the model state from the checkpoint specified in `checkpoint_path` within the JSON file and continue training from that point:

```bash
python cifar100.py --resume XXXXX.json
```

For more details about configuring checkpoints, refer to the `checkpoint_path` section in the **Parameters Description**.

### Customizing Training

To customize the training process, modify the parameters in the JSON file and rerun the script. You can configure settings such as the model architecture, optimizer, learning rate, batch size, and training scheduler to explore their impact on performance. Experimenting with these parameters will help you understand their influence on the convergence behavior.

### Training on Tiny ImageNet

To train a model on **Tiny ImageNet**, use the tiny_imagenet.py script. The process is straightforward and follows the same principles as training on other datasets. Examples are provided below:

```bash
python tiny_imagenet.py XXXXX.json
python tiny_imagenet.py --resume XXXXX.json
```

Add the --resume option to continue training from a previously saved checkpoint. The checkpoint path should be specified in the checkpoint_path parameter within the JSON file.
