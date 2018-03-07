# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import os

'''unsupervised learning -  Generative Adversarial Networks'''
def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

def Data_Processing(batch_size):

    '''In this Gan tutorial, we don't need the label data.'''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data': to2d(train_img)}, batch_size=batch_size, shuffle=True)  # training data
    return train_iter,len(train_img)

def Generator():
    '''
    <structure> is based on "" Generative Adversarial Networks paper
    authored by Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio


    #I refer to the following
    with reference to the below sentense
    We trained adversarial nets an a range of datasets including MNIST[21], the Toronto Face Database
    (TFD) [27], and CIFAR-10 [19]. The generator nets used a mixture of rectifier linear activations [17,
    8] and sigmoid activations, while the discriminator net used maxout [9] activations. Dropout [16]
    was applied in training the discriminator net. While our theoretical framework permits the use of
    dropout and other noise at intermediate layers of the generator, we used noise as the input to only
    the bottommost layer of the generator network.
    '''

    #generator neural networks
    noise = mx.sym.Variable('noise') # The size of noise is 100.

    g_affine1 = mx.sym.FullyConnected(data=noise, name='g_affine1', num_hidden=256)
    generator1= mx.sym.Activation(data=g_affine1, name='g_sigmoid1', act_type='relu')

    g_affine2 = mx.sym.FullyConnected(data=generator1, name='g_affine2', num_hidden=512)
    generator2= mx.sym.Activation(data=g_affine2, name='g_sigmoid2', act_type='relu')

    g_affine3 = mx.sym.FullyConnected(data=generator2, name='g_affine3', num_hidden=784)
    g_out= mx.sym.Activation(data=g_affine3, name='g_sigmoid3', act_type='sigmoid')
    return g_out

def Discriminator():

    zero_prevention=1e-12
    #discriminator neural networks
    data = mx.sym.Variable('data') # The size of data is 784(28*28)

    d_affine1 = mx.sym.FullyConnected(data=data,name = 'd_affine1' , num_hidden=500)
    discriminator1 = mx.sym.Activation(data=d_affine1, name='d_sigmoid1', act_type='relu')
    discriminator1=mx.sym.Dropout(data=discriminator1,p=0.3,name='drop_out_1')

    d_affine2 = mx.sym.FullyConnected(data=discriminator1,name = 'd_affine2' , num_hidden=100)
    discriminator2 = mx.sym.Activation(data=d_affine2, name='d_sigmoid2', act_type='relu')
    discriminator2 = mx.sym.Dropout(data=discriminator2,p=0.3,name='drop_out_2')

    d_affine3 = mx.sym.FullyConnected(data=discriminator2, name='d_affine3', num_hidden=1)
    d_out = mx.sym.Activation(data=d_affine3, name='d_sigmoid3', act_type='sigmoid')

    '''expression-1'''
    #out1 = mx.sym.MakeLoss(mx.symbol.log(d_out),grad_scale=-1.0,normalization='batch',name="loss1")
    #out2 = mx.sym.MakeLoss(mx.symbol.log(1.0-d_out),grad_scale=-1.0,normalization='batch',name='loss2')

    '''expression-2,
    question? Why multiply the loss equation by -1?
    answer : for Maximizing the Loss function , and This is because mxnet only provides optimization techniques that minimize.
    '''
    '''
    Why two 'losses'?
    To make it easier to implement than the reference.
    '''
    out1 = mx.sym.MakeLoss(-1.0*mx.symbol.log(d_out+zero_prevention),grad_scale=1.0,normalization='batch',name="loss1")
    out2 = mx.sym.MakeLoss(-1.0*mx.symbol.log(1.0-d_out+zero_prevention),grad_scale=1.0,normalization='batch',name='loss2')

    group=mx.sym.Group([out1,out2])

    return group

def GAN(epoch,noise_size,batch_size,save_period,load_weights):

    train_iter,train_data_number= Data_Processing(batch_size)
    #No need, but must be declared.
    label =mx.nd.zeros((batch_size,))

    '''
    Generative Adversarial Networks

    <structure>
    generator(size = 128) - 256 - 512 - (size = 784 : image generate)

    discriminator(size = 784) - 500 - 100 - (size=1 : Identifies whether the image is an actual image or not)

    cost_function - MIN_MAX cost_function
    '''
    '''Network'''

    generator=Generator()
    discriminator=Discriminator()

    '''In the code below, the 'inputs_need_grad' parameter in the 'mod.bind' function is very important.'''

    # =============module G=============
    modG = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))
    modG.bind(data_shapes=[('noise', (batch_size, noise_size))], label_shapes=None, for_training=True)

    #load the saved modG data
    G_weights_path="weights/modG-{}.params".format(load_weights)

    if os.path.exists(G_weights_path) :
        print("Load Generator weights")
        modG.load_params(G_weights_path)
    else :
        modG.init_params(initializer=mx.initializer.Normal(sigma=0.01))

    modG.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.0002})

    # =============module discriminator[0],discriminator[1]=============
    modD_0 = mx.mod.Module(symbol=discriminator[0], data_names=['data'], label_names=None, context= mx.gpu(0))
    modD_0.bind(data_shapes=train_iter.provide_data,label_shapes=None,for_training=True,inputs_need_grad=True)

    #load the saved modD_0 data
    D_weights_path="weights/modD_0-{}.params".format(load_weights)
    if os.path.exists(D_weights_path) :
        print("Load Discriminator weights")
        modD_0.load_params(D_weights_path)
    else :
        modD_0.init_params(initializer=mx.initializer.Normal(sigma=0.01))

    modD_0.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.0002})

    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).

    In here, for sharing the Discriminator parameters, we must to use shared_module=modD_0
    """
    modD_1 = mx.mod.Module(symbol=discriminator[1], data_names=['data'], label_names=None, context= mx.gpu(0))
    modD_1.bind(data_shapes=train_iter.provide_data,label_shapes=None,for_training=True,inputs_need_grad=True,shared_module=modD_0)

    # =============generate image=============
    column_size=10 ; row_size=10
    test_mod = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))
    test_mod.bind(data_shapes=[mx.io.DataDesc(name='noise', shape=(column_size*row_size,noise_size))],label_shapes=None,shared_module=modG,for_training=False,grad_req='null')

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

    ####################################training loop############################################
    # =============train===============
    for epoch in range(1,epoch+1,1):
        Max_cost_0 = 0
        Max_cost_1 = 0
        Min_cost = 0
        total_batch_number = np.ceil(train_data_number / (batch_size * 1.0))
        train_iter.reset()
        for batch in train_iter:

            modG.forward(data_batch=mx.io.DataBatch(data=[mx.random.normal(loc=0.0, scale=0.1, shape=(batch_size,noise_size))],label=None), is_train=True)
            modG_output = modG.get_outputs()

            ################################updating only parameters related to modD.########################################
            # update discriminator on noise data
            '''MAX : modD_1 : cost : (-mx.symbol.log(1-discriminator2))  - noise data Discriminator update , bigger and bigger -> smaller and smaller discriminator2'''

            modD_1.forward(data_batch=mx.io.DataBatch(data=modG_output,label=None), is_train=True)

            '''Max_Cost of noise data Discriminator'''
            Max_cost_1+=modD_1.get_outputs()[0].asnumpy().astype(np.float32)

            modD_1.backward()
            modD_1.update()

            # updating discriminator on real data

            '''MAX : modD_0 : cost: (-mx.symbol.log(discriminator2)) real data Discriminator update , bigger and bigger discriminator2'''
            modD_0.forward(data_batch=batch, is_train=True)

            '''Max_Cost of real data Discriminator'''
            Max_cost_0+=modD_0.get_outputs()[0].asnumpy().astype(np.float32)
            modD_0.backward()
            modD_0.update()


            ################################updating only parameters related to modG.########################################
            # update generator on noise data
            '''MIN : modD_0 : cost : (-mx.symbol.log(discriminator2)) - noise data Discriminator update  , bigger and bigger discriminator2'''
            modD_0.forward(data_batch=mx.io.DataBatch(data=modG_output, label=None), is_train=True)
            modD_0.backward()

            '''Max_Cost of noise data Generator'''
            Min_cost+=modD_0.get_outputs()[0].asnumpy().astype(np.float32)

            diff_v = modD_0.get_input_grads()
            modG.backward(diff_v)

            modG.update()

        Max_C=(Max_cost_0+Max_cost_1)/total_batch_number*1.0
        Min_C=Min_cost/total_batch_number*1.0

        # cost print
        print("epoch : {}".format(epoch))
        print("Max Discriminator Cost : {}".format(Max_C.mean()))
        print("Min Generator Cost : {}".format(Min_C.mean()))

        # weights save
        if not os.path.exists("weights"):
            os.makedirs("weights")

        #Save the data
        if epoch % save_period == 0:

            print('Saving weights')
            modG.save_params("Weights/modG-{}.params" .format(epoch))
            modD_0.save_params("Weights/modD_0-{}.params"  .format(epoch))

            """
            Parameters
            shared_module : Module
                Default is `None`. This is used in bucketing. When not `None`, the shared module
                essentially corresponds to a different bucket -- a module with different symbol
                but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
            """

            '''test_method-2'''
            test_mod.forward(data_batch=mx.io.DataBatch(data=[mx.random.normal(loc=0.0, scale=0.1, shape=(column_size*row_size,noise_size))],label=None))
            result = test_mod.get_outputs()[0]
            result = result.asnumpy()

            '''range adjustment 0 ~ 1 -> 0 ~ 255 '''
            result = result * 255.0

            '''generator image visualization'''
            fig, ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
            fig.suptitle('generator')
            for j in range(row_size):
                for i in range(column_size):
                    ax[j][i].set_axis_off()
                    ax[j][i].imshow(np.reshape(result[i + j * column_size], (28, 28)), cmap='gray')

            if not os.path.exists("Generate_Image"):
                os.makedirs("Generate_Image")

            fig.savefig("Generate_Image/generator_Epoch_{}.png".format(epoch))
            plt.close(fig)
    print("Optimization complete.")

    #################################Generating Image####################################
    '''load method1 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''Annotate only when running test data. and Uncomment only if it is 'load method1' or 'load method2'''
    #test_mod.set_params(arg_params=arg_params, aux_params=aux_params)

    '''test_method-1'''
    '''
    noise = noise_iter.next()
    test_mod.forward(noise, is_train=False)
    result = test_mod.get_outputs()[0]
    result = result.asnumpy()
    print np.shape(result)
    '''
    '''load method2 - using the shared_module'''
    test_mod.forward(data_batch=mx.io.DataBatch(data=[mx.random.normal(loc=0.0, scale=0.1, shape=(column_size*row_size,noise_size))],label=None))
    result = test_mod.get_outputs()[0]
    result = result.asnumpy()

    '''range adjustment 0 ~ 1 -> 0 ~ 255 '''
    result = result * 255.0

    '''generator image visualization'''
    fig, ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig.suptitle('generator')
    for j in range(row_size):
        for i in range(column_size):
            ax[j][i].set_axis_off()
            ax[j][i].imshow(np.reshape(result[i + j * column_size], (28, 28)), cmap='gray')

    if not os.path.exists("Generate_Image"):
        os.makedirs("Generate_Image")

    fig.savefig("Generate_Image/Final_generator.png")
    plt.show()


if __name__ == "__main__":
    print("GAN_starting in main")
    GAN(epoch=100, noise_size=128, batch_size=128, save_period=100,load_weights=100)
else:
    print("GAN_imported")
