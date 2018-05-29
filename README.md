# bcmlp

## Info
**BCMLP: Binary-connected Multilayer Perceptrons**<br>
Ningqi Luo', Binheng Song", Yinxu Pan', and Bin Shen'.<br>
'Department of Computer Science, "Graduate School at Shenzhen<br>
Tsinghua University.<br>
Email: lnq16@mails.tsinghua.edu.cn<br>
Submitted to 25th International Conference on Neural Information Processing, [ICONIP'2018](https://conference.cs.cityu.edu.hk/iconip/)

## Introduction
Sparse connection has been used both to reduce network complexity and sensitivity to input perturbations in multilayer perceptrons as well as artificial neural networks. We propose a novel binary-connected multilayer perceptrons where arbitrary node is connected with the only two nodes of previous layer. The sensitivity of this model is discussed both in theoretical methods and simulation experiments. Comparisons with related works show that our scheme achieves the least amount of parameters, the lowest deviation to input perturbations, and the highest accuracy in the noisy classification task.

## How to Use
### Prerequisites
  - Python 3.6.4
  - numpy 1.14.0
  - Tensorflow 1.4.0

### Files
  - bcmlp.py: the proposed model
  - sensitivity.py: for calculating the BCMLP's sensitivity
  - Exp*.py and funcs.py: for simulation experiments
  - model structure and exp reslut are show in ./pics 

### Experiment
  - Run ExpSensitivity.py
  - Run ExpClassification.py

## Related work
Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958 (2014). [Github](https://github.com/mdenil/dropout)<br>
Locally_connected_layer, Keras Documentation. [Keras](http://keras-cn.readthedocs.io/en/latest/layers/locally_connected_layer/)<br>

## Acknowledgment
We gratefully acknowledge the support from department of computer science, Tsinghua University. This work is supported by Tsinghua Multimedia Lab-SZ.
