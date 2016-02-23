# ild-cnn
This is supplementary material for the manuscript: 

>"Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network"  
M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou  
IEEE Transactions on Medical Imaging (2016)

### Environment:
We used this code on a Linux machine with Ubuntu (14.04.3 LTS) using the following setup:  
- CUDA (7.5)
- cuDNN (v3)
- python (2.7.6) with  
  * theano (0.8)
  * keras (0.3.2)
  * numpy (1.10.4)
  * argparse (1.2.1)
  * sklearn (0.17)
- python-opencv (2.4.10)

### Component Description:
There are three major components
- `main.py`      : the main script which parses the train parameters, loads some sample data and runs the training of the CNN.
- `helpers.py`    : a file with some helper functions for parsing the input parameters, loading sample data and calculating a number of evaluation metrics
- `cnn_model.py`  : this file implements the architecture of the proposed CNN and trains it.

### How to use:
`python main.py` [run with the default parameters]  
`python main.py -h` [for help]
