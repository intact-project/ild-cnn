'''
This is a part of the supplementary material uploaded along with 
the manuscript:

    "Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network"
    M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou
    IEEE Transactions on Medical Imaging (2016)
    http://dx.doi.org/10.1109/TMI.2016.2535865

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

For more information please read the README file. The files can also 
be found at: https://github.com/intact-project/ild-cnn
'''

import sys
import cv2
import numpy as np
import helpers as H
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU

# debug
from ipdb import set_trace as bp

def get_FeatureMaps(L, policy, constant=17):
    return {
        'proportional': (L+1)**2,
        'static': constant,
    }[policy]

def get_Obj(obj):
    return {
        'mse': 'MSE',
        'ce': 'categorical_crossentropy',
    }[obj]

def get_model(input_shape, output_shape, params):

    print('compiling model...')
        
    # Dimension of The last Convolutional Feature Map (eg. if input 32x32 and there are 5 conv layers 2x2 fm_size = 27)
    fm_size = input_shape[-1] - params['cl']
    
    # Tuple with the pooling size for the last convolutional layer using the params['pf']
    pool_siz = (np.round(fm_size*params['pf']).astype(int), np.round(fm_size*params['pf']).astype(int))
    
    # Initialization of the model
    model = Sequential()

    # Add convolutional layers to model
    model.add(Convolution2D(params['k']*get_FeatureMaps(1, params['fp']), 2, 2, init='orthogonal', input_shape=input_shape[1:]))
    model.add(LeakyReLU(params['a']))
    for i in range(2, params['cl']+1):
        model.add(Convolution2D(params['k']*get_FeatureMaps(i, params['fp']), 2, 2, init='orthogonal'))
        model.add(LeakyReLU(params['a']))
        
    # Add Pooling and Flatten layers to model
    if params['pt'] == 'Avg':
        model.add(AveragePooling2D(pool_size=pool_siz))
    elif params['pt'] == 'Max':
        model.add(MaxPooling2D(pool_size=pool_siz))
    else:
        sys.exit("Wrong type of Pooling layer")
    model.add(Flatten())
    model.add(Dropout(params['do']))

    # Add Dense layers and Output to model
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6, init='he_uniform'))
    model.add(LeakyReLU(0))
    model.add(Dropout(params['do']))
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*2, init='he_uniform'))
    model.add(LeakyReLU(0))
    model.add(Dropout(params['do']))
    model.add(Dense(output_shape[1], init='he_uniform', activation='softmax'))

    # Compile model and select optimizer and objective function
    if params['opt'] not in ['Adam', 'Adagrad', 'SGD']:
        sys.exit('Wrong optimizer: Please select one of the following. Adam, Adagrad, SGD')
    if get_Obj(params['obj']) not in ['MSE', 'categorical_crossentropy']:
        sys.exit('Wrong Objective: Please select one of the following. MSE, categorical_crossentropy')
    model.compile(optimizer=params['opt'], loss=get_Obj(params['obj']))

    return model

def train(x_train, y_train, x_val, y_val, params):
    ''' TODO: documentation '''

    
    # Parameters String used for saving the files
    parameters_str = str('_d' + str(params['do']).replace('.', '') +
                         '_a' + str(params['a']).replace('.', '') + 
                         '_k' + str(params['k']).replace('.', '') + 
                         '_c' + str(params['cl']).replace('.', '') + 
                         '_s' + str(params['s']).replace('.', '') + 
                         '_pf' + str(params['pf']).replace('.', '') + 
                         '_pt' + params['pt'] +
                         '_fp' + str(params['fp']).replace('.', '') +
                         '_opt' + params['opt'] +
                         '_obj' + params['obj'])

    # Printing the parameters of the model
    print('[Dropout Param] \t->\t'+str(params['do']))
    print('[Alpha Param] \t\t->\t'+str(params['a']))
    print('[Multiplier] \t\t->\t'+str(params['k']))
    print('[Patience] \t\t->\t'+str(params['patience']))
    print('[Tolerance] \t\t->\t'+str(params['tolerance']))
    print('[Input Scale Factor] \t->\t'+str(params['s']))
    print('[Pooling Type] \t\t->\t'+ params['pt'])
    print('[Pooling Factor] \t->\t'+str(str(params['pf']*100)+'%'))
    print('[Feature Maps Policy] \t->\t'+ params['fp'])
    print('[Optimizer] \t\t->\t'+ params['opt'])
    print('[Objective] \t\t->\t'+ get_Obj(params['obj']))
    print('[Results filename] \t->\t'+str(params['res_alias']+parameters_str+'.txt'))

    # Rescale Input Images
    if params['s'] != 1:
        print('\033[93m'+'Rescaling Patches...'+'\033[0m')
        x_train = np.asarray(np.expand_dims([cv2.resize(x_train[i, 0, :, :], (0,0), fx=params['s'], fy=params['s']) for i in xrange(x_train.shape[0])], 1))
        x_val = np.asarray(np.expand_dims([cv2.resize(x_val[i, 0, :, :], (0,0), fx=params['s'], fy=params['s']) for i in xrange(x_val.shape[0])], 1))
        print('\033[92m'+'Done, Rescaling Patches'+'\033[0m')
        print('[New Data Shape]\t->\tX: '+str(x_train.shape))

    model = get_model(x_train.shape, y_train.shape, params)

    # Counters-buffers
    maxf         = 0
    maxacc       = 0
    maxit        = 0
    maxtrainloss = 0
    maxvaloss    = np.inf
    p            = 0
    it           = 0
    best_model   = model

    # Open file to write the results
    open(params['res_alias']+parameters_str+'.csv', 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')
    open(params['res_alias']+parameters_str+'-Best.csv', 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')
    
    while p < params['patience']:
        p += 1

        # Fit the model for one epoch
        print('Epoch: ' + str(it))
        history = model.fit(x_train, y_train, batch_size=128, nb_epoch=1, validation_data=(x_val,y_val), shuffle=True)

        # Evaluate models
        y_score = model.predict(x_val, batch_size=1050)
        fscore, acc, cm = H.evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))
        print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))

        # Write results in file
        open(params['res_alias']+parameters_str+'.csv', 'a').write(str(str(it)+', '+str(fscore)+', '+str(acc)+', '+str(np.max(history.history['loss']))+', '+str(np.max(history.history['val_loss']))+'\n'))

        # check if current state of the model is the best and write evaluation metrics to file
        if fscore > maxf*params['tolerance']:  # if fscore > maxf*params['tolerance']:
            p            = 0  # restore patience counter
            best_model   = model  # store current model state
            maxf         = fscore 
            maxacc       = acc
            maxit        = it
            maxtrainloss = np.max(history.history['loss'])
            maxvaloss    = np.max(history.history['val_loss'])

            print(np.round(100*cm/np.sum(cm,axis=1).astype(float)))
            open(params['res_alias']+parameters_str+'-Best.csv', 'a').write(str(str(maxit)+', '+str(maxf)+', '+str(maxacc)+', '+str(maxtrainloss)+', '+str(maxvaloss)+'\n'))

        it += 1

    print('Max: fscore:', maxf, 'acc:', maxacc, 'epoch: ', maxit, 'train loss: ', maxtrainloss, 'validation loss: ', maxvaloss)

    return best_model

