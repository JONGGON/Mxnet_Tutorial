# -*- coding: utf-8 -*-
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

        '''
        <very important>
        # mean of [:]  : This sets the contents of the array instead of setting the array to a new value not overwriting the variable.
        # For more information, see reference
        '''
        net.arg_dict["data"][:] = batch.data[0]
        net.forward()
        '''
        net.outputs[0] : [batch_size,1, 10, 16, 1] .
        '''
        output=net.outputs[0]
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
        '''
        <very important>
        # mean of [:]  : This sets the contents of the array instead of setting the array to a new value not overwriting the variable.
        # For more information, see reference
        '''
        net.arg_dict["data"][:] = batch.data[0]
        net.arg_dict["label"][:] = batch.label[0]
        net.forward()

    '''
    net.outputs[1] : [batch_size,784] .
    '''
    data = net.arg_dict["data"].asnumpy() * 255
    reconstruction_out = net.outputs[0].asnumpy() * 255

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

    # (1) Get the name of the 'argument'
    arg_names = output_list[0].list_arguments()

    #caustion!!! in hear, need label's shape
    arg_shapes, output_shapes, aux_shapes = output_list[0].infer_shape(data=(batch_size,1,28,28), label=(batch_size,))

    # (2) Make space for 'argument' - mutable type - If it is declared as below, it is kept in memory.
    arg_dict = dict(zip(arg_names, [mx.nd.random.normal(loc=0, scale=0.01, shape=shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict= dict(zip(arg_names[1:-1], [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes[1:-1]])) #Exclude input output

    aux_args = [mx.nd.zeros(shape=shape, ctx=ctx) for shape in aux_shapes]

    if epoch==0 and graphviz==True:
        if reconstruction:
            total_loss = mx.viz.plot_network(symbol=output_list[0], shape={"data": (batch_size,1,28,28),"label" : (batch_size,)})
            total_loss.view("total_loss")
        else :
            margin_loss = mx.viz.plot_network(symbol=output_list[0], shape={"data": (batch_size,1,28,28),"label" : (batch_size,)})
            margin_loss.view("margin_loss")

    if reconstruction: #reconstruction=True
        if os.path.exists("weights/MNIST_Reconstruction_weights-{}.param".format(load_period)):
            print("MNIST_Reconstruction_weights-{}.param exists".format(load_period))
            pretrained = mx.nd.load("weights/MNIST_Reconstruction_weights-{}.param".format(load_period))
            for name in arg_names:
                if name == "data" or name == "label":
                    continue
                else:
                    arg_dict[name] = pretrained[name]
        else:
            print("weight initialization")

    else: #reconstruction=False
        if os.path.exists("weights/MNIST_weights-{}.param".format(load_period)):
            print("MNIST_weights-{}.param exists".format(load_period))
            pretrained = mx.nd.load("weights/MNIST_weights-{}.param".format(load_period))
            for name in arg_names:
                if name == "data" or name == "label":
                    continue
                else:
                    arg_dict[name] = pretrained[name]
        else:
            print("weight initialization")

    network=output_list[0].bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='write' , aux_states=aux_args)

    if reconstruction:
        capsule_output = output_list[1].bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='null' , aux_states=aux_args, shared_exec=network)
        reconstruction_output = output_list[2].bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='null' , aux_states=aux_args, shared_exec=network)
    else:
        capsule_output = output_list[1].bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='null' , aux_states=aux_args, shared_exec=network)

    #optimizer
    state=[]
    optimizer = mx.optimizer.Adam(learning_rate=0.001)

    for shape in arg_shapes[1:-1]:
        state.append(optimizer.create_state(0,mx.nd.zeros(shape=shape,ctx=ctx)))

    if not os.path.exists("weights"):
        os.makedirs("weights")

    # learning
    for i in tqdm(range(1,epoch+1,1)):
        '''
        In the paper,'including the exponentially decaying learning rate', But
        In this implementation, Multiply the learning_rate by 0.99 for every 10 steps.
        '''
        if i%10==0:
            optimizer.set_learning_rate(0.001*pow(0.99,i))

        train_iter.reset()
        for batch in train_iter:
            '''
            <very important>
            # mean of [:]  : This sets the contents of the array instead of setting the array to a new value not overwriting the variable.
            # For more information, see reference
            '''
            arg_dict["data"][:] = batch.data[0]
            arg_dict["label"][:] = batch.label[0]
            out=network.forward()
            network.backward(out)

            for j,name in enumerate(arg_names[1:-1]):
                optimizer.update(0, arg_dict[name] , grad_dict[name] , state[j])

        if  reconstruction:
            print("epoch : {}, last total loss : {}".format(i,mx.nd.mean(network.outputs[0]).asscalar()))
            if i % save_period == 0:

                mx.nd.save("weights/MNIST_Reconstruction_weights-{}.param".format(i), arg_dict)
        else:
            print("epoch : {}, last margin loss : {}".format(i,mx.nd.mean(network.outputs[0]).asscalar()))
            if i % save_period == 0:
                mx.nd.save("weights/MNIST_weights-{}.param".format(i),arg_dict)

        test_accuracy = evaluate_accuracy(test_iter, capsule_output)
        print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

    print("#Optimization complete\n")

    test_accuracy = evaluate_accuracy(test_iter, capsule_output)
    print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))
    if reconstruction:
        generate_image(test_iter, reconstruction_output)

if __name__ == "__main__":
    CapsNet(reconstruction=True, epoch=1, batch_size=128, save_period=100, load_period=100, ctx=mx.gpu(0),graphviz=False)
else:
    print("imported")

