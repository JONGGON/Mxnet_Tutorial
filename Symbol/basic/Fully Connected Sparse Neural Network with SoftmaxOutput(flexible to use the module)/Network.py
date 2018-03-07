# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
import os
from tqdm import *
logging.basicConfig(level=logging.INFO)

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

    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter = mx.io.NDArrayIter(data={'data' : to2d(test_img)}, label={'label' : test_lbl}) #test data

    '''neural network'''
    data = mx.sym.Variable('data' , stype='csr')
    label = mx.sym.Variable('label')
    weight1 = mx.symbol.Variable('1_weight', stype='row_sparse', shape=(784, 100))
    bias1 = mx.symbol.Variable('1_bias', shape=(100,))
    weight2 = mx.symbol.Variable('2_weight', stype='row_sparse', shape=(100, 10))
    bias2 = mx.symbol.Variable('2_bias', shape=(10,))


    with mx.name.Prefix("Sparse_FNN_"):
        # first_hidden_layer
        output = mx.sym.broadcast_add(mx.sym.sparse.dot(data, weight1), bias1)
        output = mx.sym.sparse.relu(output)
        output = mx.sym.broadcast_add(mx.sym.sparse.dot(output, weight2), bias2)

    output=mx.sym.SoftmaxOutput(data=output,label=label)

    shape = {"data": (batch_size,784)}
    graph=mx.viz.plot_network(symbol=output,shape=shape)
    if epoch==1:
        graph.view()
    print(output.list_arguments())

    # training mod
    mod = mx.mod.Module(symbol=output, data_names=['data'], label_names=['label'], context=mx.gpu(0))
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    #load the saved mod data
    weights_path="weights/mod-{}.params".format(load_weights)

    if os.path.exists(weights_path) :
        print("Load weights")
        mod.load_params(weights_path)
    else :
        mod.init_params(initializer=mx.initializer.Normal(sigma=0.01))

    mod.init_optimizer(optimizer='rmsprop',optimizer_params={'learning_rate': 0.001})

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
            mod.save_params("weights/mod-{}.params" .format(epoch))

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
    '''test'''
    result = test.predict(test_iter).asnumpy().argmax(axis=1)
    print('Final accuracy : {}%' .format(float(sum(test_lbl == result)) / len(result)*100.0))

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100)
else:
    print("NeuralNet_imported")

