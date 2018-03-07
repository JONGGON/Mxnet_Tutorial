# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import os
from tqdm import *

'''unsupervised learning -  Autoencoder'''

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

def NeuralNet(epoch,batch_size,save_period,load_weights):
    '''
    load_data

    1. SoftmaxOutput must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data

    2. LogisticRegressionOutput , LinearRegressionOutput , MakeLoss and so on.. must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data
    '''

    '''In this Autoencoder tutorial, we don't need the label data.'''
    (_, _, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (_, _, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter  = mx.io.NDArrayIter(data={'input' : to2d(train_img)},label={'input_' : to2d(train_img)}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'input' : to2d(test_img)},label={'input_' : to2d(test_img)}) #test data

    '''Autoencoder network

    <structure>
    input - encode - middle - decode -> output
    '''
    input = mx.sym.Variable('input')
    output= mx.sym.Variable('input_')

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
    result=mx.sym.LinearRegressionOutput(data=result ,label=output)

    shape = {"input": (batch_size,784)}
    graph=mx.viz.plot_network(symbol=result,shape=shape)
    if epoch==1:   
        graph.view()
    print(result.list_arguments())
    
    #training mod
    mod = mx.mod.Module(symbol=result, data_names=['input'],label_names=['input_'], context=mx.gpu(0))
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    weights_path="weights/mod-{}.params".format(load_weights)

    if os.path.exists(weights_path) : 
        print("Load weights")
        mod.load_params(weights_path)
    else :
        #mod.init_params(initializer=mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))
        mod.init_params(initializer=mx.initializer.Uniform(scale=1))

    #mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})
    mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    #test mod
    test = mx.mod.Module(symbol=result, data_names=['input'],label_names=['input_'], context=mx.gpu(0))
    '''load method2 - using the shared_module'''
    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    """
    test.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label,shared_module=mod,for_training=False)

    # Network information print
    print(mod.data_names)
    print(mod.label_names)
    print(train_iter.provide_data)
    print(train_iter.provide_label)


    '''############Although not required, the following code should be declared.#################'''

    '''make evaluation method 1 - Using existing ones.
        metrics = {
        'acc': Accuracy,
        'accuracy': Accuracy,
        'ce': CrossEntropy,
        'f1': F1,
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'top_k_accuracy': TopKAccuracy
    }'''

    metric = mx.metric.create(['acc','mse'])

    '''make evaluation method 2 - Making new things.'''
    '''
    Custom evaluation metric that takes a NDArray function.
    Parameters:
    •feval (callable(label, pred)) – Customized evaluation function.
    •name (str, optional) – The name of the metric.
    •allow_extra_outputs (bool) – If true, the prediction outputs can have extra outputs.
    This is useful in RNN, where the states are also produced in outputs for forwarding.
    '''
    def zero(label, pred):
        return 0

    null = mx.metric.CustomMetric(zero)

    for epoch in tqdm(range(1,epoch+1,1)):
        print("epoch : {}".format(epoch))
        train_iter.reset()
        #total_batch_number = np.ceil(len(train_img) / (batch_size * 1.0))
        #temp=0
        for batch in train_iter:

            mod.forward(batch, is_train=True)
            mod.backward()
            mod.update()

            #cost
            temp=(mod.get_outputs()[0].asnumpy()-batch.data[0].asnumpy())
            cost = np.sum(0.5*np.square(temp),axis=1).mean()

        #print("training_data : {}".format(mod.score(train_iter, ['mse'])))
        print("last cost value : {}".format(cost))

        if not os.path.exists("weights"):
            os.makedirs("weights")

        #Save the data
        if epoch%save_period==0:
            print('Saving weights')
            mod.save_params("weights/mod-{}.params" .format(epoch))

    # Network information print
    #print(mod.data_shapes)
    #print(mod.label_shapes)
    #print(mod.output_shapes)
    #print(mod.get_params())
    #print(mod.get_outputs())
    print("Optimization complete.")

    #################################TEST####################################
    '''load method2 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''Annotate only when running test data. and Uncomment only if it is 'load method2' '''
    #test.set_params(arg_params, aux_params)

    '''test'''
    column_size=10 ; row_size=10 #batch_size <= column_size x row_size <= 10000

    result = test.predict(test_iter,num_batch=column_size*row_size).asnumpy()

    '''range adjustment 0 ~ 1 -> 0 ~ 255 '''
    result = result*255.0

    '''generator image visualization'''
    fig_g ,  ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
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
            ax_r[j][i].imshow(test_img[i+j*column_size], cmap='gray')
    fig_r.savefig("real.png")

    plt.show()

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100)
else:
    print("NeuralNet_imported")
