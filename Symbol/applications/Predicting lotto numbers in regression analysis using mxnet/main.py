# -*- coding: utf-8 -*-
import mxnet as mx
import data_preprocessing as dp
from Network import LottoNet

'''implement'''
net=LottoNet(epoch=0,batch_size=50,save_period=10000,load_period=60000)
print(net[0])
print(net[1])