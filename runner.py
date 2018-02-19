# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:35:00 2018

@author: wzhu16
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ResNet as resnet
import utils
import os
from imageio import imread
import pickle


im2 = utils.load_image224('Strawberry.jpg')
batch1 = im2
batch = [batch1]

resnet = resnet.resnet()

resnet.build_model()

pb2 = resnet.predict(batch)   
utils.print_prob(pb2[0], 'imagenet-classes.txt')
