# -*- coding: utf-8 -*-
import mxnet as mx
from Network import NeuralNet

'''implement'''

NeuralNet(epoch=1,batch_size=256,save_period=500,load_weights=500, ctx=mx.gpu(0))
