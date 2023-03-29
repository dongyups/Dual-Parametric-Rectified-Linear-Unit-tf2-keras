# Dual-Parametric-Rectified-Linear-Unit-tf2-keras
Dual Parametric ReLU (DPReLU) is based on ReLU which has two different trainable parameters above or below zero.

### Required
This code is tested with python==3.8, tensorflow-gpu==2.7, and cuda/cudnn 11.2/8.1
- Python 3.X
- Tensorflow 2.X
- CUDA 10.X or 11.X for gpu settings depends on your hardware

### How to use
```python3
from tf2dprelu import DPReLU

previous_layer = ... # ex) conv, dense, ...
dprelu_activated = DPReLU(previous_layer)
```
