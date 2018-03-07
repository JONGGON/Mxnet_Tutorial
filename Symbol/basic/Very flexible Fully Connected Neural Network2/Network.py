# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
from tqdm import *
import os
logging.basicConfig(level=logging.INFO)

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

def NeuralNet(epoch,batch_size,save_period,load_weights,ctx=mx.gpu(0)):


    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter  = mx.io.NDArrayIter(data={'data' : to2d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data

    '''neural network'''
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    with mx.name.Prefix("FNN_"):

        # hidden_layer
        affine1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=100)
        hidden1 = mx.sym.Activation(data=affine1, name='relu1', act_type="relu")

        # output_layer
        output_affine = mx.sym.FullyConnected(data=hidden1, name='fc2', num_hidden=10)

    output=mx.sym.SoftmaxOutput(data=output_affine,label=label)

    # (1) Get the name of the 'argument'
    arg_names = output.list_arguments()
    arg_shapes, output_shapes, aux_shapes = output.infer_shape(data=(batch_size,784))

    # (2) Make space for 'argument' - mutable type - If it is declared as below, it is kept in memory.
    arg = [mx.nd.random_normal(loc=0, scale=0.01, shape=shape, ctx=ctx) for shape in arg_shapes]
    grad = [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes] #Exclude input output

    shape = {"data": (batch_size,784)}
    graph=mx.viz.plot_network(symbol=output,shape=shape)
    if epoch==1:
        graph.view()

    if os.path.exists("weights/MNIST_weights-{}.param".format(load_weights)):
        print("MNIST_weights-{}.param exists".format(load_weights))
        pretrained = mx.nd.load("weights/MNIST_weights-{}.param".format(load_weights))

        for i,name in enumerate(arg_names):
            if name == "data" or name == "label":
                continue
            else:
                arg[i] = pretrained[i]
    else:
        print("weight initialization")


    '''You only need to bind once.
    With bind, you can change the argument values of args and args_grad. The value of a list or dictionary is mutable.
    See the code below for an approach.'''
    network=output.bind(ctx=ctx, args=arg, args_grad=grad, grad_req='write')

    #optimizer
    state=[]
    optimizer = mx.optimizer.Adam(learning_rate=0.001)
    '''
    Creates auxiliary state for a given weight.
    Some optimizers require additional states, e.g. as momentum, in addition to gradients in order to update weights. 
    This function creates state for a given weight which will be used in update. This function is called only once for each weight.
    '''
    for shape in arg_shapes[1:-1]:
        state.append(optimizer.create_state(0,mx.nd.zeros(shape=shape,ctx=ctx)))

    #learning
    for i in tqdm(range(1,epoch+1,1)):
        print("epoch : {}".format(i))
        train_iter.reset()
        for batch in train_iter:
            '''
            <very important>
            # mean of [:]  : This sets the contents of the array instead of setting the array to a new value not overwriting the variable.
            # For more information, see reference
            '''
            #batch.data[0].copyto(arg[0])
            #batch.label[0].copyto(arg[-1])
            arg[0][:]= batch.data[0]
            arg[-1][:] = batch.label[0]
            network.forward(is_train=True)
            network.backward()

            for j in range(len(arg_names[1:-1])):
                optimizer.update(0, arg[j+1] , grad[j+1] , state[j])

        result = network.outputs[0].argmax(axis=1)
        print('Training batch accuracy : {}%'.format((float(sum(arg[-1].asnumpy() == result.asnumpy())) / len(result.asnumpy()))*100))

        if not os.path.exists("weights"):
            os.makedirs("weights")

        if i%save_period==0:
            mx.nd.save("weights/MNIST_weights-{}.param".format(i), arg)

    print("#Optimization complete\n")

    #test
    for batch in test_iter:

        #batch.data[0].copyto(arg[0])
        #batch.label[0].copyto(arg[-1])
        arg[0][:]= batch.data[0]
        arg[-1][:] = batch.label[0]
        network.forward()

    result = network.outputs[0].argmax(axis=1)
    print("###########################")
    print('Test batch accuracy : {}%'.format((float(sum(arg[-1].asnumpy() == result.asnumpy())) / len(result.asnumpy())) * 100))

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100,ctx=mx.gpu(0))
else:
    print("NeuralNet_imported")

