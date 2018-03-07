# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
from capsule import *
import data_download as dd
import logging
from tqdm import *
import os
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255.0

#evaluate the datz
def evaluate_accuracy(data_iterator , net):

    data_iterator.reset() # It must be written.
    numerator = 0
    denominator = 0

    for batch in data_iterator:

        net.forward(batch)
        output=net.get_outputs()[0]
        output=output.square().sum(axis=3, keepdims=True).sqrt()
        output=output.reshape((-1,10))
        output=output.argmax(axis=1)# (batch_size , num_outputs)

        output=output.asnumpy()
        label=batch.label[0].asnumpy()
        numerator += sum(output == label)
        denominator += output.shape[0]

    return (numerator / denominator)

def generate_image(data_iterator , net):

    data_iterator.reset() # It must be written.
    for batch in data_iterator:
        net.forward(batch)
        output=net.get_outputs()[0]

    data = batch.data[0].asnumpy() * 255.0
    reconstruction_out = output.asnumpy() * 255.0

    '''test'''
    column_size=8 ; row_size=8 #     column_size x row_size <= 10000

    print("show image")
    '''Reconstruction image visualization'''

    if not os.path.exists("Reconstruction_Image"):
        os.makedirs("Reconstruction_Image")

    fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_g.suptitle('MNIST_generator')
    for j in range(row_size):
        for i in range(column_size):
            ax_g[j][i].set_axis_off()
            ax_g[j][i].imshow(np.reshape(reconstruction_out[i + j * column_size],(28,28)),cmap='gray')
    fig_g.savefig("Reconstruction_Image/MNIST_Reconstruction.png")

    '''real image visualization'''
    fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_r.suptitle('MNIST_real')
    for j in range(row_size):
        for i in range(column_size):
            ax_r[j][i].set_axis_off()
            ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
    fig_r.savefig("Reconstruction_Image/MNIST_real.png")

    plt.show()

def CapsNet(reconstruction,epoch,batch_size,save_period,load_period,ctx=mx.gpu(0), graphviz=False):

    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''
    In the paper,'Training is performed on 28? 28 MNIST images have been shifted by up to 2 pixels in each direction with zero padding', But
    In this implementation, the original data is not transformed as above.
    '''
    '''data loading referenced by Data Loading API '''

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True, last_batch_handle='roll_over') #training data
    test_iter  = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size , shuffle=False ,last_batch_handle='roll_over') #test data

    '''
    reconstruction=true  
    output_list[0] -> total_loss=margin_loss+reconstruction_loss
    output_list[1] -> capsule_output
    output_list[2] -> reconstruction_output
    
    reconstruction=False
    output_list[0] -> margin_loss
    output_list[1] -> capsule_output
    '''
    output_list = capsule(reconstruction=reconstruction, routing_iteration=1, batch_size=batch_size)

    # training mod
    network = mx.mod.Module(symbol=output_list[0], data_names=['data'], label_names=['label'], context=ctx)
    network.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label, for_training=True)

    if epoch == 0 and graphviz == True:

        if reconstruction:
            total_loss = mx.viz.plot_network(symbol=output_list[0], shape={"data": (batch_size,1,28,28),"label" : (batch_size,)})
            total_loss.view("total_loss")
        else :
            margin_loss = mx.viz.plot_network(symbol=output_list[0], shape={"data": (batch_size,1,28,28),"label" : (batch_size,)})
            margin_loss.view("margin_loss")

    if reconstruction: #reconstruction==True
        if os.path.exists("weights/MNIST_Reconstruction_weights-{}.param".format(load_period)):
            print("MNIST_Reconstruction_weights-{}.param exists".format(load_period))
            network.load_params("weights/MNIST_Reconstruction_weights-{}.param".format(load_period))
        else:
            print("weight initialization")
            network.init_params(initializer=mx.initializer.Normal(sigma=0.1))

    else: #reconstruction=False
        if os.path.exists("weights/MNIST_weights-{}.param".format(load_period)):
            print("MNIST_weights-{}.param exists".format(load_period))
            network.load_params("weights/MNIST_weights-{}.param".format(load_period))
        else:
            print("weight initialization")
            network.init_params(initializer=mx.initializer.Normal(sigma=0.1))

    if reconstruction:
        capsule_output = mx.mod.Module(symbol=output_list[1], data_names=['data'], label_names=None, context=ctx)
        reconstruction_output = mx.mod.Module(symbol=output_list[2], data_names=['data'], label_names=['label'], context=ctx)

        capsule_output.bind(data_shapes=test_iter.provide_data, label_shapes=None, for_training=False, shared_module=network, grad_req='null')
        reconstruction_output.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label, for_training=False, shared_module=network, grad_req='null')
    else:
        capsule_output = mx.mod.Module(symbol=output_list[1], data_names=['data'], label_names=None, context=ctx)
        capsule_output.bind(data_shapes=test_iter.provide_data, label_shapes=None, for_training=False, shared_module=network, grad_req='null')

    lr_sch = mx.lr_scheduler.FactorScheduler(step=5000, factor=0.99)
    network.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001, 'lr_scheduler' : lr_sch })

    if not os.path.exists("weights"):
        os.makedirs("weights")

    # learning
    for i in tqdm(range(1,epoch+1,1)):

        train_iter.reset()
        for batch in train_iter:
            network.forward(batch)
            out_grads=network.get_outputs()
            network.backward(out_grads=out_grads)
            network.update()

        if reconstruction:
            print("epoch : {}, last total loss : {}".format(i,mx.nd.mean(network.get_outputs()[0]).asscalar()))
            if i % save_period == 0:
                print('Saving weights')
                network.save_params("weights/MNIST_Reconstruction_weights-{}.param".format(i))
        else:
            print("epoch : {}, last margin loss : {}".format(i,mx.nd.mean(network.get_outputs()[0]).asscalar()))
            if i % save_period == 0:
                print('Saving weights')
                network.save_params("weights/MNIST_weights-{}.param".format(i))

        test_accuracy = evaluate_accuracy(test_iter, capsule_output)
        print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

    print("Optimization complete\n")

    test_accuracy = evaluate_accuracy(test_iter, capsule_output)
    print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

    if reconstruction:
        generate_image(test_iter, reconstruction_output)

if __name__ == "__main__":
    CapsNet(reconstruction=True, epoch=1, batch_size=128, save_period=100, load_period=100, ctx=mx.gpu(0),graphviz=False)
else:
    print("imported")

