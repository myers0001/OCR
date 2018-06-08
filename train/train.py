#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload 
import densenet
import io

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#by zh
#img_h = 32
#img_w = 280
img_h = 32
img_w = 280
batch_size = 64
maxlabellength = 10

def get_session(gpu_fraction=1.0):  
  
    num_threads = os.environ.get('OMP_NUM_THREADS')
    #by zh,1.0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_options.allow_growth = True
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines() 
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    #dic '58491.jpg': ['2', '2', '5', '9', '11', '15', '8', '13', '3', '11']
    return dic

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize
        
        return r_n  

def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    #print _imagefile '23186.jpg', '11380.jpg', '78634.jpg', '61548.jpg', '53030.jpg', '74030.jpg',
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])
    
    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        # print shufimagefile 9110.jpg' '16277.jpg' '35680.jpg' '16571.jpg' '20968.jpg' '78623.jpg'
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img,axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            #print str ['7', '4', '12', '14', '15', '8', '3', '7', '9', '6']
            label_length[i] = len(str) 
            
            if(len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[int(i), :len(str)] =[int(i) - 1 for i in str]
            #print labels
            #print ('x=',x)  [[[[0.43333334],[0.43725491],[0.44117647],...,[0.44901961],[0.42156863],[0.38235295]
            #print ('labels=',labels) [3.0e+00, 7.0e+00, 8.0e+00, 4.0e+00, 5.0e+00, 1.2e+01, 4.0e+00
            #print ('input_length=',input_length) [35.]
            #print ('label_length=',label_length) [10.]

        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }

        outputs = {'ctc': np.zeros([batchsize])} 
        yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) 

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    char_set = io.open('char_std_zh.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    #add by zh
    nclass = len(char_set) +1

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = './models/pretrain_model/keras.h5'
    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen('data_train.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen('data_test.txt', './images', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='./models/weights-densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4**epoch
    #learning_rate = np.array([lr_schedule(i) for i in range(10)]) by zh
    learning_rate = 0.00005
    #changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader, 
    	#steps_per_epoch = 3607567 // batch_size,by zh
        steps_per_epoch= (100000 - 8500) // batch_size,
        #steps_per_epoch=None,
    	epochs = 4,
    	initial_epoch = 0, 
    	validation_data = test_loader,
        #validation_steps = None,
        validation_steps= 8500 // batch_size,
    	# validation_steps = (1000 - 896 )36440 // batch_size, by zh
    	callbacks = [checkpoint, earlystop, changelr, tensorboard])

