import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import *
import os

''' Autoencoder '''

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#MNIST dataset
def MNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    # last_batch  "discard" : last_batch must be a discard or rollover , The batchsize of 'test_data' must be greater than "column_size x row_size".
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,128, shuffle=False, last_batch="discard") #Loads data from a dataset and returns mini-batches of data.
    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    # last_batch  "discard" : last_batch must be a discard or rollover , The batchsize of 'test_data' must be greater than "column_size x row_size".
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,128 , shuffle=False , last_batch="discard") #Loads data from a dataset and returns mini-batches of data.
    return train_data , test_data


#evaluate the data
def generate_image(data_iterator , num_inputs , network , ctx , dataset):

    for data, label in data_iterator:

        data = data.as_in_context(ctx).reshape((-1,num_inputs))
        output = network(data,0.0) # when test , 'Dropout rate' must be 0.0
        data = data.asnumpy() * 255.0
        output=output.asnumpy() * 255.0

    '''test'''
    column_size=10 ; row_size=10 #     column_size x row_size <= 10000

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
def Autoencoder(epoch = 100 , batch_size=10, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate= 0.1 , dataset = "MNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset =="MNIST":
        train_data , test_data = MNIST(batch_size)
    elif dataset == "FashionMNIST":
        train_data, test_data = FashionMNIST(batch_size)
    else:
        return "The dataset does not exist."

    # data structure

    num_inputs = 784

    #network parameter

    #encoder
    num_hidden1 = 200
    num_hidden2 = 100

    #decoder
    num_hidden1_ = 100
    num_hidden2_= 200

    num_outputs = num_inputs

    if dataset == "MNIST":
        path = "weights/MNIST_weights-{}".format(load_period)
    elif dataset == "FashionMNIST":
        path = "weights/FashionMNIST_weights-{}".format(load_period)

    if os.path.exists(path):
        print("loading weights")
        [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5]= nd.load(path)  # weights load

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

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5]
    else:
        print("initializing weights")

        with ctx:
            W1 = nd.random.normal(loc=0, scale=0.2, shape=(num_hidden1,num_inputs))
            B1 = nd.random.normal(loc=0, scale=0.2, shape=num_hidden1)

            W2 = nd.random.normal(loc=0, scale=0.2, shape=(num_hidden2,num_hidden1))
            B2 = nd.random.normal(loc=0, scale=0.2, shape=num_hidden2)

            W3 = nd.random.normal(loc=0, scale=0.2, shape=(num_hidden1_, num_hidden2))
            B3 = nd.random.normal(loc=0, scale=0.2, shape=num_hidden1_)

            W4 = nd.random.normal(loc=0, scale=0.2, shape=(num_hidden2_, num_hidden1_))
            B4 = nd.random.normal(loc=0, scale=0.2, shape=num_hidden2_)

            W5 = nd.random.normal(loc=0, scale=0.2, shape=(num_outputs, num_hidden2_))
            B5 = nd.random.normal(loc=0, scale=0.2, shape=num_outputs)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5]

    # attach gradient!!!
    for i, param in enumerate(params):
        param.attach_grad()
    
    #Fully Neural Network with 1 Hidden layer 
    def network(X,dropout=0.0):
        
        #encoder
        H1 = nd.Activation(nd.FullyConnected(data=X , weight=W1 , bias=B1 , num_hidden=num_hidden1), act_type="sigmoid")
        H1 = nd.Dropout(data=H1 , p=dropout) # apply dropout layer!!!
        H2 = nd.Activation(nd.FullyConnected(data=H1 , weight=W2 , bias=B2 , num_hidden=num_hidden2), act_type="sigmoid")
        H2 = nd.Dropout(data=H2 , p=dropout) # apply dropout layer!!!

        #decoder
        H3 = nd.Activation(nd.FullyConnected(data=H2 , weight=W3 , bias=B3 , num_hidden=num_hidden1_), act_type="sigmoid")
        H3 = nd.Dropout(data=H3 , p=dropout) # apply dropout layer!!!
        H4 = nd.Activation(nd.FullyConnected(data=H3 , weight=W4 , bias=B4 , num_hidden=num_hidden2_), act_type="sigmoid")
        H4 = nd.Dropout(data=H4 , p=dropout) # apply dropout layer!!!
        H5 = nd.Activation(nd.FullyConnected(data=H4 , weight=W5 , bias=B5 , num_hidden=num_outputs), act_type="sigmoid")
        out = H5
        return out

    def MSE(output, label):
        #return - nd.sum(label * nd.log(output), axis=1)
        return nd.sum(nd.square(output-label),axis=1)

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
            data = data.as_in_context(ctx).reshape((-1,num_inputs))

            with autograd.record():
                output = network(data,0.2)

                #loss definition
                loss = MSE(output,data) # (batch_size,)
                cost = nd.mean(loss).asscalar()

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
            if dataset=="MNIST":
                nd.save("weights/MNIST_weights-{}".format(i),params)
            elif dataset=="FashionMNIST":
                nd.save("weights/FashionMNIST_weights-{}".format(i),params)

    #show image
    generate_image(test_data , num_inputs , network , ctx  ,dataset)

    return "optimization completed"

if __name__ == "__main__":
    Autoencoder(epoch=100, batch_size=128, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="MNIST", ctx=mx.gpu(0))
else :
    print("Imported")


