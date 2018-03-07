# -*- coding: utf-8 -*-
import mxnet as mx
import data_download as dd
import logging
import os
logging.basicConfig(level=logging.INFO)
 
def NeuralNet(epoch,batch_size,save_period,load_weights):

    time_step=28
    rnn_hidden_number = 100
    layer_number = 1
    fc_number= 100
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
            cell.add(mx.rnn.FusedRNNCell(num_hidden=rnn_hidden_number, num_layers=1, bidirectional=False, mode="gru", prefix="gru_{}".format(i), params=None, forget_bias=1.0, get_next_state=True))
            if Dropout_rate > 0 and (layer_number-1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="gru_dropout_{}".format(i)))
        else:
            cell.add(mx.rnn.GRUCell(num_hidden=rnn_hidden_number, prefix="gru_{}".format(i)))
            if Dropout_rate > 0 and (layer_number - 1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="gru_dropout_{}".format(i)))

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
    if epoch==1 and use_cudnn: #why? use_cudnn : for simple graph
        graph.view()
    print(output.list_arguments())

    # training mod
    mod = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=mx.gpu(0))
    # test mod
    test = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=mx.gpu(0))

    # Network information print
    print(mod.data_names)
    print(mod.label_names)
    print(train_iter.provide_data)
    print(train_iter.provide_label)

    '''if the below code already is declared by mod.fit function, thus we don't have to write it.
    but, when you load the saved weights, you must write the below code.'''
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    # weights save
    if not os.path.exists("weights"):
        os.makedirs("weights")

    if use_cudnn:
        model_name = 'weights/Fused_Neural_Net'
        checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)
    else : 
        model_name = 'weights/Neural_Net'
        checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)

    #weights load
    # When you want to load the saved weights, uncomment the code below.
    weights_path= model_name+"-0{}.params".format(load_weights)
    if os.path.exists(weights_path):
        print("Load weights")
        symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, load_weights)
        #the below code needs mod.bind, but If arg_params and aux_params is set in mod.fit, you do not need the code below, nor do you need mod.bind.
        mod.set_params(arg_params, aux_params)

    mod.fit(train_iter, initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),
            optimizer='adam',
            optimizer_params={'learning_rate': 0.001},
            eval_metric=mx.metric.MSE(),
            # Once the loaded parameters are declared here,You do not need to declare mod.set_params,mod.bind
            num_epoch=epoch,
            arg_params=None,
            aux_params=None,
            epoch_end_callback=checkpoint)


    # Network information print
    print(mod.data_shapes)
    print(mod.label_shapes)
    print(mod.output_shapes)
    print(mod.get_params())
    print(mod.get_outputs())
    print("training_data : {}".format(mod.score(train_iter, ['mse', 'acc'])))
    print("Optimization complete.")

    #################################TEST####################################
    '''load method1 - load the saved parameter'''
    #symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 100)

    '''load method2 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''load method3 - using the shared_module'''
    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    """
    test.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label,shared_module=mod,for_training=False)

    '''Annotate only when running test data. and Uncomment only if it is 'load method1' or 'load method2' '''
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
