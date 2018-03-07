# -*- coding: utf-8 -*-
import mxnet as mx
from capsule_network import CapsNet

'''implement'''
CapsNet(reconstruction=True, epoch=10, batch_size=200, save_period=10, load_period=10, ctx=mx.gpu(0), graphviz=True)
