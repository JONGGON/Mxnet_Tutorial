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

def variational_autoencoder(n_latent=16, batch_size=128, generator = False):

    input = mx.sym.Variable('data')

    with mx.name.Prefix("variational Autoencoder"):
        # encode
        affine1 = mx.sym.FullyConnected(data=input,name='encode1',num_hidden=100)
        encode1 = mx.sym.Activation(data=affine1, name='relu1', act_type="relu")

        # encode
        affine2 = mx.sym.FullyConnected(data=encode1, name='encode2', num_hidden=100)
        encode2 = mx.sym.Activation(data=affine2, name='relu2', act_type="relu")

        # encode
        affine3 = mx.sym.FullyConnected(data=encode2, name='encode3', num_hidden=n_latent*2)

        #mean, variance
        mu_logvar = mx.sym.split(affine3, axis=1, num_outputs=2)
        mu = mu_logvar[0]
        log_var = mu_logvar[1]

        zero_mean_Gaussian = mx.sym.random_normal(loc=0, scale=1, shape=(batch_size, n_latent))
        z = mu + mx.sym.exp(0.5 * log_var) * zero_mean_Gaussian

        if generator:
            z = mx.sym.Variable('latent_vector')
            affine4 = mx.sym.FullyConnected(data=z, name='decode1', num_hidden=100)
        else:
            affine4 = mx.sym.FullyConnected(data=z, name='decode1', num_hidden=100)

        # decode
        decode1 = mx.sym.Activation(data=affine4, name='relu3', act_type="relu")

        # decode
        affine5 = mx.sym.FullyConnected(data=decode1,name='decode2',num_hidden=100)
        decode2 = mx.sym.Activation(data=affine5, name='relu4', act_type="relu")

        # output
        result = mx.sym.FullyConnected(data=decode2, name='result', num_hidden=784)
        result = mx.sym.Activation(data=result, name='sigmoid', act_type="sigmoid")

        if generator:
            return result
        else:
            # reconstruction loss - cross entropy
            log_loss = mx.sym.sum(input * mx.sym.log(result + 1e-12) + (1 - input) * mx.sym.log(1 - result + 1e-12), axis=1)

            # KL Divergence Regularizer - Just accept it.
            KL = 0.5 * mx.sym.sum(mx.sym.exp(log_var) + mu * mu - 1 - log_var, axis=1)

            # ELBO = Evidence Lower Bound
            ELBO = log_loss - KL

            #LogisticRegressionOutput contains a sigmoid function internally. and It should be executed with xxxx_lbl_one_hot data.
            return mx.sym.LinearRegressionOutput(data=-ELBO)


def NeuralNet(epoch,batch_size,save_period,load_weights,ctx=mx.gpu(0)):


    (_, _, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (_, _, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter  = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'label' : to2d(train_img)}, batch_size=batch_size, shuffle=True) #training data

    #network
    n_latent = 16
    output=variational_autoencoder(n_latent=n_latent, batch_size=batch_size, generator=False)
    generator=variational_autoencoder(n_latent=n_latent, batch_size=batch_size, generator=True)

    # (1) Get the name of the 'argument'
    arg_names = output.list_arguments()
    arg_shapes, output_shapes, aux_shapes = output.infer_shape(data=(batch_size,784))

    # (2) Make space for 'argument' - mutable type - If it is declared as below, it is kept in memory.
    arg_dict = dict(zip(arg_names, [mx.nd.random_normal(loc=0, scale=0.01, shape=shape, ctx=ctx) for shape in arg_shapes]))
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

    network = output.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='write')
    #optimizer
    state=[]
    optimizer = mx.optimizer.Adam(learning_rate=0.001)


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
            arg_dict["data"][:] = batch.data[0]
            network.forward(is_train=True)
            network.backward()

            for j,name in enumerate(arg_names[1:-1]):
                optimizer.update(0, arg_dict[name] , grad_dict[name] , state[j])

        print('Training cost : {}%'.format(mx.nd.mean(network.outputs[0]).asscalar()))
        if not os.path.exists("weights"):
            os.makedirs("weights")

        if i%save_period==0:
            print('Saving weights')
            mx.nd.save("weights/MNIST_weights-{}.param".format(i), arg_dict)

    print("Optimization complete\n")

    '''test'''
    column_size=10 ; row_size=10 #  batch_size <= column_size x row_size
    normal_distribution = mx.nd.random_normal(loc=0, scale=1, shape=(column_size*row_size, n_latent), ctx=ctx)
    arg_dict['latent_vector'] = normal_distribution

    generator = generator.bind(ctx=ctx, args=arg_dict)
    generator.forward()
    '''range adjustment 0 ~ 1 -> 0 ~ 255 '''
    result = generator.outputs[0].asnumpy()*255.0

    '''generator image visualization'''
    fig_g , ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_g.suptitle('generator')
    for j in range(row_size):
        for i in range(column_size):
            ax_g[j][i].set_axis_off()
            ax_g[j][i].imshow(np.reshape(result[i+j*column_size],(28,28)),cmap='gray')

    fig_g.savefig("generator.png")
    plt.show()

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100,ctx=mx.gpu(0))
else:
    print("NeuralNet_imported")

