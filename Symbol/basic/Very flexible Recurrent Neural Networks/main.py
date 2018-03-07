# -*- coding: utf-8 -*-
import mxnet as mx
from Network import NeuralNet

'''implement'''

NeuralNet(epoch=1,batch_size=128,save_period=50,load_weights=50, ctx=mx.gpu(0))
