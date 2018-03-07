# -*- coding: utf-8 -*-
import mxnet as mx
from capsule_network import CapsNet

'''
Implementing 'Dynamic Routing Between Capsules' Using Gluon
Note : Gpu memry usage is 5GB
'''

'''implement'''
CapsNet(reconstruction=True, epoch=0, batch_size=128, save_period=10, load_period=10, ctx=mx.gpu(0), graphviz=True)
