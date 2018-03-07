# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
from tqdm import *
import os
logging.basicConfig(level=logging.INFO)

def NeuralNet(epoch,batch_size,save_period,load_weights,ctx=mx.gpu(0)):

    time_step=28
    rnn_hidden_number = 300
    layer_number = 1
    fc_number=100
    class_number=10
    Dropout_rate=0.2
    use_cudnn = False

    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : train_img},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter  = mx.io.NDArrayIter(data={'data' : test_img}, label={'label' : test_lbl}, batch_size=batch_size) #test data

    '''neural network'''
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    data = mx.sym.transpose(data, axes=(1, 0, 2))  # (time,batch,column)

    cell = mx.rnn.SequentialRNNCell()

    for i in range(layer_number):
        if use_cudnn:
            cell.add(mx.rnn.FusedRNNCell(num_hidden=rnn_hidden_number, num_layers=1, bidirectional=False, mode="lstm", prefix="lstm_{}".format(i), params=None, forget_bias=1.0, get_next_state=True))
            if Dropout_rate > 0 and (layer_number-1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="lstm_dropout_{}".format(i)))
        else:
            cell.add(mx.rnn.LSTMCell(num_hidden=rnn_hidden_number, prefix="lstm_{}".format(i)))
            if Dropout_rate > 0 and (layer_number - 1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="lstm_dropout_{}".format(i)))

    # if you see the unroll function
    output, state = cell.unroll(length=time_step, inputs=data, merge_outputs=False, layout='TNC')
    '''FullyConnected Layer'''
    affine1 = mx.sym.FullyConnected(data=output[-1], num_hidden=fc_number, name='affine1')
    act1 = mx.sym.Activation(data=affine1, act_type='relu', name='relu1')
    affine2 = mx.sym.FullyConnected(data=act1, num_hidden=class_number, name = 'affine2')
    output = mx.sym.SoftmaxOutput(data=affine2, label=label, name='softmax')

    # (1) Get the name of the 'argument'
    arg_names = output.list_arguments()
    arg_shapes, output_shapes, aux_shapes = output.infer_shape(data=(batch_size,28,28))

    # (2) Make space for 'argument' - mutable type - If it is declared as below, it is kept in memory.
    arg_dict = dict(zip(arg_names, [mx.nd.random_normal(loc=0, scale=0.02, shape=shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict= dict(zip(arg_names[1:-1], [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes[1:-1]])) #Exclude input output

    shape = {"data": (batch_size,28,28)}
    graph=mx.viz.plot_network(symbol=output,shape=shape)
    if epoch==1 and use_cudnn:
        graph.view("Fused")
    print(output.list_arguments())


    if use_cudnn and os.path.exists("weights/MNIST_Fused_weights-{}.param".format(load_weights)):
        print("MNIST_Fused_weights-{}.param exists".format(load_weights))
        pretrained = mx.nd.load("weights/MNIST_Fused_weights-{}.param".format(load_weights))
        for name in arg_names:
            if name == "data" or name == "label":
                continue
            else:
                arg_dict[name] = pretrained[name]
    elif use_cudnn==False and os.path.exists("weights/MNIST_weights-{}.param".format(load_weights)):
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

        result = network.outputs[0].argmax(axis=1)
        print('Training batch accuracy : {}%'.format((float(sum(arg_dict["label"].asnumpy() == result.asnumpy())) / len(result.asnumpy()))*100))

        if not os.path.exists("weights"):
            os.makedirs("weights")

        if i%save_period==0:
            print('Saving weights')
            if use_cudnn:
                mx.nd.save("weights/MNIST_Fused_weights-{}.param".format(i), arg_dict)
            else :
                mx.nd.save("weights/MNIST_weights-{}.param".format(i), arg_dict)

    print("#Optimization complete\n")

    #test
    for batch in test_iter:
        arg_dict["data"][:] = batch.data[0]
        arg_dict["label"][:] = batch.label[0]
        network.forward()

    result = network.outputs[0].argmax(axis=1)
    print("###########################")
    print('Test batch accuracy : {}%'.format((float(sum(arg_dict["label"].asnumpy() == result.asnumpy())) / len(result.asnumpy())) * 100))

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100,ctx=mx.gpu(0))
else:
    print("NeuralNet_imported")

