# ild-cnn
This is supplementary material for the manuscript: 

>"Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network"  
M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou  
IEEE Transactions on Medical Imaging (2016)  

In case of any questions, please do not hesitate to contact us.

### Environment:
This code was used on a Linux machine with Ubuntu (14.04.3 LTS) using the following setup:  
- CUDA (7.5)
- cuDNN (v3)
- python (2.7.6) with  
  * [Theano](https://github.com/Theano/Theano) (0.8)
  * [keras](https://github.com/fchollet/keras) (0.3.2)
  * [numpy](https://github.com/numpy/numpy) (1.10.4)
  * [argparse](https://github.com/bewest/argparse) (1.2.1)
  * [scikit-learn](https://github.com/scikit-learn/scikit-learn) (0.17)
- python-opencv (2.4.10)

### Component Description:
There are three major components
- `main.py`      : the main script which parses the train parameters, loads some sample data and runs the training of the CNN.
- `helpers.py`    : a file with some helper functions for parsing the input parameters, loading sample data and calculating a number of evaluation metrics
- `cnn_model.py`  : this file implements the architecture of the proposed CNN and trains it.

### How to use:
`python main.py` : runs an experiment with the default parameters  
`python main.py -h` : shows the help message

### Output Description:
The execution outputs two csv formatted files with the performance metrics of the CNN. The first contains the performances for each training epoch while the second only for the epochs that improved the performance. The code prints the same output while running as well as a confusion matrix every time the CNN performance improves.

### Disclaimer:
Copyright (C) 2016  Marios Anthimopoulos, Stergios Christodoulidis, Stavroula Mougiakakou / University of Bern  


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
