# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:56:16 2018

@author: wzhu16
"""

import tensorflow as tf
import numpy as np
import copy
import utils
import os
MEAN = [103.939, 116.779, 123.68]

class resnet(object):
    def __init__(self):
        path = os.getcwd()
        path = os.path.join(path, "res50.npy")
        self.data_dict = np.load(path, encoding='latin1').item()
        self.sess = tf.Session()



    def normal_img(self, rgb):
        rgb_scaled = rgb * 255.0
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - MEAN[0],
            green - MEAN[1],
            red - MEAN[2],
        ])
        return bgr
       
    def build_model(self):

        
        ##
#        start_time = time.time()
        self.rgb = tf.placeholder("float32", [None, None, None, 3])
#        self.y_true = tf.placeholder("float32", [None,None,None, NUM_CLASSES])
        bgr = self.normal_img(self.rgb)        
        ##
        
        ## conv1
        self.conv1 = self.conv_block_1(bgr)

        ## conv2
        self.conv2 = self._build_box(self.conv1, 3, '2')        

        ## conv3
        self.conv3 = self._build_box(self.conv2, 4, '3')
        
        ## conv4
        self.conv4 = self._build_box(self.conv3, 6, '4')

        ## conv5
        self.conv5 = self._build_box(self.conv4, 3, '5')
        
        self.pool5 = tf.nn.avg_pool(self.conv5, [1,7,7,1], [1,1,1,1], 'VALID', name='pool5')
        self.pool5_resize = tf.reshape(self.pool5, [-1,2048])
        self.shortcut = self.pool5
        self.fc1000 = self.fc_1000(self.pool5_resize)
        
        
        self.prob = tf.nn.softmax(self.fc1000, name = 'softmax_prob')
    
    
        init = tf.global_variables_initializer()

        self.sess.run(init)

    def conv_block_1(self,x):
        filters = tf.Variable(self.data_dict['conv1']['weights'], name = 'conv1_weight')
        bias = tf.Variable(self.data_dict['conv1']['biases'], name = 'conv1_bias')
        conv = tf.nn.bias_add(tf.nn.conv2d(x, filters, [1,2,2,1],padding='SAME', name = 'conv1'), bias)
        mean, offset, scale, variance = self.get_bn_para('bn_conv1')
        bn = tf.nn.batch_normalization(conv, mean, variance, offset, scale, 1e-5, name = 'bn1')
        relu = tf.nn.relu(bn,'relu1')
        pool = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME',name = 'pool1')
        return pool

    def i2c(self,num):
        return(chr(num+97))
        

    def _build_box(self,x, num_box,layer_name):
        if layer_name == '2':
            first_stride = 1
        else:
            first_stride = 2
            
        for i in range(num_box):
            if i == 0:
                x, shortcut = self._residual_block_first(x, first_stride,[layer_name+self.i2c(i),'2a'])
                
                x = self._residual_block_middle(x, [layer_name+self.i2c(i),'2b'])
                x = self._residual_block_end(x, shortcut, True,[layer_name+self.i2c(i),'2c'])
                
            else:
                x, shortcut = self._residual_block_first(x, 1,[layer_name+self.i2c(i),'2a'])
                self.shortcut = shortcut
                x = self._residual_block_middle(x, [layer_name+self.i2c(i),'2b'])
                x = self._residual_block_end(x, shortcut, False,[layer_name+self.i2c(i),'2c'])
   
        
        return x

    def _residual_block_first(self, x, stride,pos):
        name = pos[0]+pos[1]
        x_ident = x
        
        
        with tf.variable_scope(name) as scope:
            # Residual
            x = self.conv_layer(x, stride, pos)
            x = self.bn_layer(x, pos)
            x = tf.nn.relu(x, name=name+'relu')
        return x, x_ident

    def _residual_block_middle(self, x, pos):
        name = pos[0]+pos[1]
        with tf.variable_scope(name) as scope:
            x = self.conv_layer(x, 1, pos)
            x = self.bn_layer(x, pos)
            x = tf.nn.relu(x, name=name+'relu')
        return x
    
    def _residual_block_end(self, x, x_ident,resize, pos):
        name = pos[0]+pos[1]
        pos_ident = copy.copy(pos)
        pos_ident[1] = '1'
        with tf.variable_scope(name) as scope:
            x = self.conv_layer(x, 1, pos)
            x = self.bn_layer(x, pos)
            
            if resize is True:
                
                if pos_ident[0] == '2a':
                    x_id_stride = 1
                else:
                    x_id_stride = 2
                print('stride for identity', x_id_stride)
                x_id = self.conv_layer(x_ident, x_id_stride, pos_ident)
                
                x_id = self.bn_layer(x_id, pos_ident)
                print('resized')
            else:
                print('Use identity')
                x_id = x_ident
            
            x = x + x_id
            x = tf.nn.relu(x, name = name+'relu')
            
        return x

    
    def get_conv_filter(self, pos_tag):
        print(pos_tag)
        return tf.Variable(self.data_dict[pos_tag]['weights'], name=pos_tag)

    def conv_layer(self, x, strides = 1, pos=None,name =None):
        pos_tag = 'res'+pos[0]+'_branch'+pos[1]
        with tf.variable_scope(pos_tag):
            filt = self.get_conv_filter(pos_tag)
            conv = tf.nn.conv2d(x, filt, [1, strides, strides, 1], padding='SAME', name = pos_tag)
        return conv


    def fc_1000(self,x):
        w = tf.Variable(self.data_dict['fc1000']['weights'], name = 'fc_w')
        b = tf.Variable(self.data_dict['fc1000']['biases'], name = 'fc_b')
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        return y

    def get_bn_para(self, pos_tag):
        mean = tf.Variable(self.data_dict[pos_tag]['mean'], name = pos_tag + 'mean')
        offset = tf.Variable(self.data_dict[pos_tag]['offset'], name = pos_tag + 'offset')
        scale = tf.Variable(self.data_dict[pos_tag]['scale'], name = pos_tag + 'scale')
        variance = tf.Variable(self.data_dict[pos_tag]['variance'], name = pos_tag + 'variance')
        return mean, offset, scale, variance

    def bn_layer(self, x, pos, name= None):
        pos_tag = 'bn' + pos[0] + '_branch' + pos[1]
        with tf.variable_scope(pos_tag):
            mean, offset, scale, variance = self.get_bn_para(pos_tag)
            bn = tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-5, name = pos_tag)
        return bn


    def predict(self, data):
        y = self.sess.run(self.prob, feed_dict={self.rgb:data})
        return y

