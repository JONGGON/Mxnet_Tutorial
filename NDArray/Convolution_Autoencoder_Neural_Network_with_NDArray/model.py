import numpy as np
import mxnet as mx
import mxnet.gluon as gluon #when using data load
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import *
import os

''' ConvolutionAutoencoder '''
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255.0, label.astype(np.float32)

#MNIST dataset
def MNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    # last_batch  "discard" : last_batch must be a discard or rollover , The batchsize of 'test_data' must be greater than "column_size x row_size".
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,128 , shuffle=False, last_batch="discard") #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    # last_batch  "discard" : last_batch must be a discard or rollover , The batchsize of 'test_data' must be greater than "column_size x row_size".
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,128 , shuffle=False, last_batch="discard") #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#evaluate the data
def generate_image(data_iterator , network , ctx , dataset ):

    for i,(data, label) in enumerate(data_iterator):

        data = data.as_in_context(ctx)
        output = network(data,0.0) # when test , 'Dropout rate' must be 0.0
        
        data = data.asnumpy() * 255.0
        output=output.asnumpy() * 255.0

    '''test'''
    column_size=10 ; row_size=10 #     column_size x row_size <= 10000

    data = data.transpose(0,2,3,1)
    output = output.transpose(0,2,3,1)

    print("show image")
    '''generator image visualization'''

    if not os.path.exists("Generate_Image"):
        os.makedirs("Generate_Image")

    if dataset=="MNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('MNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Generate_Image/MNIST_generator.png")

        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('MNIST_real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
        fig_r.savefig("Generate_Image/MNIST_real.png")

    elif dataset=="FashionMNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('FashionMNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Generate_Image/FashionMNIST_generator.png")

        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('FashionMNIST_real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
        fig_r.savefig("Generate_Image/FashionMNIST_real.png")

    plt.show()


#reduce dimensions -> similar to PCA
def CNN_Autoencoder(epoch = 100 , batch_size=10, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate= 0.1 , dataset = "FashionMNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset =="MNIST":
        train_data , test_data = MNIST(batch_size)
    elif dataset == "FashionMNIST":
        train_data, test_data = FashionMNIST(batch_size)
    else:
        return "The dataset does not exist."

    if dataset == "MNIST":
        path = "weights/MNIST_weights-{}".format(load_period)
    elif dataset == "FashionMNIST":
        path = "weights/FashionMNIST_weights-{}".format(load_period)

    if os.path.exists(path):
        print("loading weights")

        [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, W6, B6, W7, B7, W8, B8] = nd.load(path)  # weights load

        W1=W1.as_in_context(ctx)
        B1=B1.as_in_context(ctx)
        W2=W2.as_in_context(ctx)
        B2=B2.as_in_context(ctx)
        W3=W3.as_in_context(ctx)
        B3=B3.as_in_context(ctx)

        W4=W4.as_in_context(ctx)
        B4=B4.as_in_context(ctx)

        W5=W5.as_in_context(ctx)
        B5=B5.as_in_context(ctx)
        W6=W6.as_in_context(ctx)
        B6=B6.as_in_context(ctx)
        W7=W7.as_in_context(ctx)
        B7=B7.as_in_context(ctx)
        W8=W8.as_in_context(ctx)
        B8=B8.as_in_context(ctx)

        params = [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, W6, B6, W7, B7, W8, B8]
    else:
        print("initializing weights")

        with ctx:
            W1 = nd.random.normal(loc=0 , scale=0.01 , shape=(60,1,3,3))
            B1 = nd.random_normal(loc=0 , scale=0.01 , shape=60)

            W2 = nd.random.normal(loc=0 , scale=0.01 , shape=(30,60,3,3))
            B2 = nd.random.normal(loc=0 , scale=0.01 , shape=30)

            W3 = nd.random.normal(loc=0 , scale=0.01 , shape=(15,30,3,3))
            B3 = nd.random.normal(loc=0 , scale=0.01 , shape=15)

            #######################################################################

            W4 = nd.random.normal(loc=0 , scale=0.01 , shape=(10,15,3,3))
            B4 = nd.random.normal(loc=0 , scale=0.01 , shape=10)

            #######################################################################

            W5 = nd.random.normal(loc=0 , scale=0.01 , shape=(15,10,3,3))
            B5 = nd.random.normal(loc=0 , scale=0.01 , shape=15)

            W6 = nd.random.normal(loc=0 , scale=0.01 , shape=(30,15,3,3))
            B6 = nd.random.normal(loc=0 , scale=0.01 , shape=30)

            W7 = nd.random.normal(loc=0 , scale=0.01 , shape=(60,30,3,3))
            B7 = nd.random.normal(loc=0 , scale=0.01 , shape=60)

            W8 = nd.random.normal(loc=0 , scale=0.01 , shape=(1,60,3,3))
            B8 = nd.random.normal(loc=0 , scale=0.01 , shape=1)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5 , W6 , B6 , W7 , B7 , W8 , B8 ]

    # attach gradient!!!
    for i, param in enumerate(params):
        param.attach_grad()
    
    '''Brief description of deconvolution.
    I was embarrassed when I first heard about deconvolution,
    but it was just the opposite of convolution.
    The formula is as follows.

    The convolution formula is  output_size = ([input_size+2*pad-kernel_size]/stride) + 1

    The Deconvolution formula is output_size = stride(input_size-1)+kernel-2*pad

    '''
    def network(X,dropout=0.0):

        #encoder
        EC_H1=nd.Activation(data= nd.Convolution(data=X , weight = W1 , bias = B1 , kernel=(3,3) , stride=(1,1)  , num_filter=60) , act_type="relu") # FashionMNIST or MNIST : result = ( batch size , 60 , 26 , 26)
        EC_H2=nd.Activation(data= nd.Convolution(data=EC_H1 , weight = W2 , bias = B2 , kernel=(3,3) , stride=(1,1) , num_filter=30), act_type="relu") # FashionMNIST or MNIST : result = ( batch size , 30 , 24 , 24)
        EC_H3=nd.Activation(data= nd.Convolution(data=EC_H2 , weight = W3 , bias = B3 , kernel=(3,3) , stride=(1,1) , num_filter=15), act_type="relu") # FashionMNIST or MNIST : result = ( batch size , 15 , 22 , 22)

        #Middle
        MC_H=nd.Activation(data= nd.Convolution(data=EC_H3 , weight = W4 , bias = B4 , kernel=(3,3) , stride=(1,1) , num_filter=10), act_type="relu") # FashionMNIST : result = ( batch size , 10 , 20 , 20)

        #decoder -  why not using Deconvolution? because NDArray.Deconvolution is not working...
        DC_H1=nd.Activation(data = nd.Convolution(data=MC_H , weight = W5 , bias = B5 , kernel=(3,3) , stride=(1,1) , pad= (2,2) , num_filter=15) , act_type="relu") # FashionMNIST or MNIST : result = ( batch size , 15 , 22 , 22)
        DC_H2=nd.Activation(data= nd.Convolution(data=DC_H1 , weight = W6 , bias = B6 , kernel=(3,3) , stride=(1,1) , pad= (2,2) , num_filter=30), act_type="relu") # FashionMNIST or MNIST  : result = ( batch size , 30 , 24 , 24)
        DC_H3=nd.Activation(data= nd.Convolution(data=DC_H2 , weight = W7 , bias = B7 , kernel=(3,3) , stride=(1,1) , pad= (2,2) , num_filter=60) , act_type="relu") # FashionMNIST or MNIST  : result = ( batch size , 60 , 26 , 26)

        #output
        out=nd.Activation(data= nd.Convolution(data=DC_H3 , weight = W8 , bias = B8 , kernel=(3,3) , stride=(1,1) , pad= (2,2), num_filter=1) , act_type="sigmoid") # FashionMNIST or MNIST : result = ( batch size , 1 , 28 , 28)

        return out

    def MSE(output, label):
        return nd.sum(0.5*nd.square(output-label) , axis=0 , exclude=True)

    #Adam optimizer
    state=[]
    optimizer=mx.optimizer.Adam(rescale_grad=1,learning_rate=learning_rate)
    for i,param in enumerate(params):
        state.append(optimizer.create_state(0,param))

    #optimizer
    def SGD(params, lr , wd , bs):
        for param in params:
             param -= ((lr * param.grad)/bs+wd*param)


    for i in tqdm(range(1,epoch+1,1)):

        for data,label in train_data:
            data = data.as_in_context(ctx)

            with autograd.record():
                output = network(data,0.0)

                #loss definition
                loss = MSE(output,data) # (batch_size,)
                cost = nd.mean(loss, axis=0).asscalar()

            loss.backward()

            for j,param in enumerate(params):
                optimizer.update(0,param,param.grad,state[j])

            #SGD(params, learning_rate , weight_decay , batch_size)

        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="FashionMNIST":
                nd.save("weights/FashionMNIST_weights-{}".format(i),params)
            elif dataset=="MNIST":
                nd.save("weights/MNIST_weights-{}".format(i),params)

    #show image
    generate_image(test_data , network , ctx ,dataset)

    return "optimization completed"

if __name__ == "__main__":
    CNN_Autoencoder(epoch=100, batch_size=128, save_period=100 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="FashionMNIST", ctx=mx.gpu(0))
else :
    print("Imported")



