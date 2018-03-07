# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import data_download as dd
import logging
import os
from tqdm import *
logging.basicConfig(level=logging.INFO)

def NeuralNet(epoch,batch_size,save_period,load_weights):

    time_step=28
    rnn_hidden_number = 200
    layer_number=1
    fc_number=100
    class_number=10
    Dropout_rate=0.2
    use_cudnn = True

    '''
    load_data

    1. SoftmaxOutput must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data
                                                                or
    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    2. LogisticRegressionOutput , LinearRegressionOutput , MakeLoss and so on.. must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    '''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : train_img},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : test_img}, label={'label' : test_lbl_one_hot}) #test data

    ####################################################-Network-################################################################
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    data = mx.sym.transpose(data, axes=(1, 0, 2))  # (time,batch,column)

    '''1. RNN cell declaration'''

    '''
    Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.
    '''
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

    '''2. Unroll the RNN CELL on a time axis.'''

    ''' unroll's return parameter
    outputs : list of Symbol
              output symbols.
    states : Symbol or nested list of Symbol
            has the same structure as begin_state()

    '''
    #if you see the unroll function
    output, state= cell.unroll(length=time_step, inputs=data, merge_outputs=False, layout='TNC')

    '''FullyConnected Layer'''
    affine1 = mx.sym.FullyConnected(data=output[-1], num_hidden=fc_number, name='affine1')
    act1 = mx.sym.Activation(data=affine1, act_type='sigmoid', name='sigmoid1')
    affine2 = mx.sym.FullyConnected(data=act1, num_hidden=class_number, name = 'affine2')
    output = mx.sym.SoftmaxOutput(data=affine2, label=label, name='softmax')


    graph=mx.viz.plot_network(symbol=output)
    if epoch==1 and use_cudnn:
        graph.view()
    print(output.list_arguments())

    # training mod
    mod = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=mx.gpu(0))
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    #load the saved mod data
    if use_cudnn:
        weights_path="weights/Fused_mod-{}.params".format(load_weights)
    else:
        weights_path="weights/mod-{}.params".format(load_weights)

    if os.path.exists(weights_path) and  use_cudnn :
        print("Fused Load weights")
        mod.load_params(weights_path)
    elif os.path.exists(weights_path) and  not use_cudnn  :
        print("Load weights")
        mod.load_params(weights_path)
    else :
        print("initializing weights")
        mod.init_params(initializer=mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))

    mod.init_params(initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1))
    mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    # test mod
    test = mx.mod.Module(symbol=output, data_names=['data'], label_names=['label'], context=mx.gpu(0))

    '''load method1 - using the shared_module'''
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
            #temp+=(mod.get_outputs()[0].asnumpy()-batch.label[0].asnumpy())

        #cost = (0.5*np.square(temp)/(total_batch_number*1.0)).mean()
        result = test.predict(test_iter).asnumpy().argmax(axis=1)
        print("training_data : {}".format(mod.score(train_iter, ['mse', 'acc'])))
        print('accuracy during learning.  : {}%'.format(float(sum(test_lbl == result)) / len(result) * 100.0))
        #print "cost value : {}".format(cost)

        if not os.path.exists("weights"):
            os.makedirs("weights")

        #Save the data
        if epoch%save_period==0:
            print('Saving weights')
            if use_cudnn:
                mod.save_params("weights/Fused_mod-{}.params".format(epoch))
            else :
                mod.save_params("weights/mod-{}.params" .format(epoch))

    # Network information print
    print(mod.data_shapes)
    print(mod.label_shapes)
    print(mod.output_shapes)
    print(mod.get_params())
    print(mod.get_outputs())

    print("Optimization complete.")
    #################################TEST####################################
    '''load method2 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''Annotate only when running test data. and Uncomment only if it is 'load method2' '''
    #test.set_params(arg_params, aux_params)

    #batch by batch accuracy
    #To use the code below, Test / batchsize must be an integer.
    '''for preds, i_batch, eval_batch in mod.iter_predict(test_iter):
        pred_label = preds[0].asnumpy().argmax(axis=1)
        label = eval_batch.label[0].asnumpy().argmax(axis=1)
        print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label == label)) / len(label)))
    '''

    result = test.predict(test_iter).asnumpy().argmax(axis=1)
    print('Final accuracy : {}%' .format(float(sum(test_lbl == result)) / len(result)*100.0))

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100)
else:
    print("NeuralNet_imported")

