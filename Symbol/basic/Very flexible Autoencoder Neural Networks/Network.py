# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
from tqdm import *
import os
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

def NeuralNet(epoch,batch_size,save_period,load_weights,ctx=mx.gpu(0)):


    (_, _, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (_, _, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter  = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'label' : to2d(train_img)}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to2d(test_img)},label={'label' : to2d(test_img)}, batch_size=batch_size) #test data

    '''Autoencoder network

    <structure>
    input - encode - middle - decode -> output
    '''
    input = mx.sym.Variable('data')
    output= mx.sym.Variable('label')

    with mx.name.Prefix("Autoencoder_"):
        # encode
        affine1 = mx.sym.FullyConnected(data=input,name='encode1',num_hidden=100)
        encode1 = mx.sym.Activation(data=affine1, name='sigmoid1', act_type="sigmoid")

        # encode
        affine2 = mx.sym.FullyConnected(data=encode1, name='encode2', num_hidden=50)
        encode2 = mx.sym.Activation(data=affine2, name='sigmoid2', act_type="sigmoid")

        # decode
        affine3 = mx.sym.FullyConnected(data=encode2, name='decode1', num_hidden=50)
        decode1 = mx.sym.Activation(data=affine3, name='sigmoid3', act_type="sigmoid")

        # decode
        affine4 = mx.sym.FullyConnected(data=decode1,name='decode2',num_hidden=100)
        decode2 = mx.sym.Activation(data=affine4, name='sigmoid4', act_type="sigmoid")

        # output
        result = mx.sym.FullyConnected(data=decode2, name='result', num_hidden=784)
        result = mx.sym.Activation(data=result, name='sigmoid5', act_type="sigmoid")

    #LogisticRegressionOutput contains a sigmoid function internally. and It should be executed with xxxx_lbl_one_hot data.
    output=mx.sym.LinearRegressionOutput(data=result ,label=output)

    # (1) Get the name of the 'argument'
    arg_names = output.list_arguments()
    arg_shapes, output_shapes, aux_shapes = output.infer_shape(data=(batch_size,784))

    # (2) Make space for 'argument' - mutable type - If it is declared as below, it is kept in memory.
    arg_dict = dict(zip(arg_names, [mx.nd.random_normal(loc=0, scale=0.02, shape=shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict= dict(zip(arg_names[1:-1], [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes[1:-1]])) #Exclude input output

    shape = {"data": (batch_size,784)}
    graph=mx.viz.plot_network(symbol=output,shape=shape)
    if epoch==1:
        graph.view()
    print(output.list_arguments())

    if os.path.exists("weights/MNIST_weights-{}.param".format(load_weights)):
        print("MNIST_weights-{}.param exists".format(load_weights))
        pretrained = mx.nd.load("weights/MNIST_weights-{}.param".format(load_weights))
        for name in arg_names:
            if name == "data" or name == "label":
                continue
            else:
                arg_dict[name] = pretrained[name]
    else:
        print("weight initialization")

    '''You only need to bind once.
    With bind, you can change the argument values of args and args_grad. The value of a list or dictionary is mutable.
    See the code below for an approach.'''
    network=output.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='write')

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
            #batch.data[0].copyto(arg_dict["data"])
            #batch.label[0].copyto(arg_dict["label"])
            arg_dict["data"][:] = batch.data[0]
            arg_dict["label"][:] = batch.label[0]
            network.forward(is_train=True)
            network.backward()

            for j,name in enumerate(arg_names[1:-1]):
                optimizer.update(0, arg_dict[name] , grad_dict[name] , state[j])

            cost = mx.nd.mean(mx.nd.divide(mx.nd.square(network.outputs[0]-arg_dict["label"]),2))

        print('Training cost : {}%'.format(cost.asscalar()))
        if not os.path.exists("weights"):
            os.makedirs("weights")

        if i%save_period==0:
            print('Saving weights')
            mx.nd.save("weights/MNIST_weights-{}.param".format(i), arg_dict)

    print("#Optimization complete\n")

    '''test'''
    column_size=10 ; row_size=10 #  batch_size <= column_size x row_size

    real=arg_dict["data"].asnumpy()
    result = network.outputs[0].asnumpy()

    '''range adjustment 0 ~ 1 -> 0 ~ 255 '''
    real = real*255.0
    result = result*255.0

    '''generator image visualization'''
    fig_g , ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_g.suptitle('generator')
    for j in range(row_size):
        for i in range(column_size):
            ax_g[j][i].set_axis_off()
            ax_g[j][i].imshow(np.reshape(result[i+j*column_size],(28,28)),cmap='gray')

    fig_g.savefig("generator.png")

    '''real image visualization'''
    fig_r ,  ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_r.suptitle('real')
    for j in range(row_size):
        for i in range(column_size):
            ax_r[j][i].set_axis_off()
            ax_r[j][i].imshow(np.reshape(real[i+j*column_size],(28,28)),cmap='gray')
    fig_r.savefig("real.png")

    plt.show()

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100,ctx=mx.gpu(0))
else:
    print("NeuralNet_imported")

