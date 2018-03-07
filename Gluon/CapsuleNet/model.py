'''
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829
Author: Jonggon Kim
'''
import numpy as np
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
from capsule import *
from tqdm import *
import os

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255.0, label.astype(np.float32)

#MNIST dataset
def MNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,batch_size , shuffle=False, last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#FashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) ,batch_size, shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,batch_size, shuffle=False, last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

def evaluate_accuracy(data_iterator , net , ctx):

    numerator = 0
    denominator = 0

    for data, label in data_iterator:

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output, reconstruction_output = net(data, label)
        '''
        output : [batch_size,1, 10, 16, 1] .
        '''
        output=output.square().sum(axis=3, keepdims=True).sqrt()
        output=output.reshape((-1,10))

        predictions = nd.argmax(output, axis=1) # (batch_size , num_outputs)
        predictions=predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += data.shape[0]

    return (numerator / denominator)

def generate_image(data_iterator , net , ctx , dataset):

    for data, label in data_iterator:

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output, reconstruction_output = net(data, label)
        data = data.asnumpy() * 255.0
        reconstruction_output=reconstruction_output.asnumpy() * 255.0

    '''test'''
    column_size=8 ; row_size=8 #     column_size x row_size <= 10000

    print("show image")
    '''Reconstruction image visualization'''

    if not os.path.exists("Reconstruction_Image"):
        os.makedirs("Reconstruction_Image")

    if dataset=="MNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('MNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(reconstruction_output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Reconstruction_Image/MNIST_Reconstruction.png")

        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('MNIST_real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
        fig_r.savefig("Reconstruction_Image/MNIST_real.png")

    elif dataset=="FashionMNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('FashionMNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(reconstruction_output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Reconstruction_Image/FashionMNIST_Reconstruction.png")

        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('FashionMNIST_real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
        fig_r.savefig("Reconstruction_Image/FashionMNIST_real.png")
    plt.show()

class Network(gluon.HybridBlock):

    def __init__(self, batch_size=None ,Routing_Iteration=None, **kwargs):

        super(Network, self).__init__(**kwargs)
        self.batch_size=batch_size
        self.Routing_Iteration=Routing_Iteration

        with self.name_scope():
            '''
            In the paper, The second layer(PrimaryCapsules) -> 8 capsules
            '''
            self.Primarycaps=Primarycaps() # (batch_size, 32, 6, 6, 8) ->(batch_size, 1152, 8)  # (batch_size, 32, 6, 6, 8) ->(batch_size, 1152, 8)
            '''
            In the paper, The third layer(DigitCaps) -> 16 capsules
            '''
            self.DigitCaps=DigitCaps(batch_size=self.batch_size)  # (batch_size, 1152, 10, 16, 1)
            self.Routing_algorithm=Routing_algorithm(Routing_Iteration=self.Routing_Iteration)  # Routing algorithm -> # (batch_size,1,10,16,1)
            self.Reconstruction_Layer=Reconstruction_Layer()

    def hybrid_forward(self, F, x, label):

        x=self.Primarycaps(x)
        x=self.DigitCaps(x)
        x=self.Routing_algorithm(x)
        reconstruction_x=self.Reconstruction_Layer(x,label)
        return x, reconstruction_x

def CapsuleNet(Reconstruction=True, epoch = 100, batch_size=256, save_period=100, load_period=100, optimizer="adam", learning_rate= 0.001, dataset = "MNIST", ctx=mx.gpu(0)):

    if dataset =="MNIST":
        '''
        In the paper,'Training is performed on 28? 28 MNIST images have been shifted by up to 2 pixels in each direction with zero padding', But
        In this implementation, the original data is not transformed as above.
        '''
        train_data , test_data = MNIST(batch_size)
        path = "weights/MNIST-{}.params".format(load_period)
    elif dataset == "FashionMNIST":
        train_data, test_data = FashionMNIST(batch_size)
        path = "weights/FashionMNIST-{}.params".format(load_period)
    else:
        return "The dataset does not exist."

    #Convolution Neural Network
    # formula : output_size=((input−weights+2*Padding)/Stride)+1
    # data size
    # MNIST, FashionMNIST = (batch size , 1 , 28 ,  28)

    # Routing_Iteration = 1 due to memory problem. It uses close to 5GB of memory.
    net = Network(batch_size=batch_size, Routing_Iteration=1)

    '''
    What you need for 'hybridize' mode.
    'DigitCaps' calculation process 'batch_size' should be specified.
    Therefore, 'batch_size' of 'test' data and 'batch_size' of 'training' data should be the same.
    '''
    net.hybridize() # for faster learning and efficient memory use

    #weights initialization
    if os.path.exists(path):
        print("loading weights")
        net.load_params(filename=path , ctx=ctx) # weights load
    else:
        print("initializing weights")
        net.collect_params().initialize(mx.init.Normal(sigma=0.01),ctx=ctx) # weights initialization
        #net.initialize(mx.init.Normal(sigma=0.1),ctx=ctx) # weights initialization

    '''
    In the paper,'including the exponentially decaying learning rate', But
    In this implementation, Multiply the learning_rate by 0.99 for every 10 steps.
    '''
    lr_scheduler=mx.lr_scheduler.FactorScheduler(step=10, factor=0.99)
    trainer = gluon.Trainer(net.collect_params() , optimizer, {"learning_rate" : learning_rate, "lr_scheduler" : lr_scheduler})

    #learning
    for i in tqdm(range(1,epoch+1,1)):
        for data , label in train_data:

            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record(train_mode=True):
                output, reconstruction_output = net(data, label)
                if Reconstruction:
                    margin_loss=Margin_Loss()(output,label)
                    recon_loss=gluon.loss.L2Loss()(reconstruction_output, data.reshape((batch_size,-1)))
                    loss=margin_loss+0.0005*recon_loss
                else:
                    loss=Margin_Loss()(output, label)

            cost=nd.mean(loss).asscalar()
            loss.backward()
            trainer.step(batch_size,ignore_stale_grad=True)

        print(" epoch : {} , last batch cost : {}".format(i, cost))
        test_accuracy = evaluate_accuracy(test_data, net, ctx)
        print("Test_acc : {0:0.3f}%".format(test_accuracy*100))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="MNIST":
                net.save_params("weights/MNIST-{}.params".format(i))

            elif dataset=="FashionMNIST":
                net.save_params("weights/FashionMNIST-{}.params".format(i))

    test_accuracy = evaluate_accuracy(test_data, net, ctx)
    print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

    if Reconstruction:
        generate_image(test_data, net, ctx, dataset)

    return "optimization completed"

if __name__ == "__main__":
    CapsuleNet(Reconstruction=True, epoch = 100, batch_size=256, save_period=100, load_period=100, optimizer="adam", learning_rate= 0.001, dataset = "MNIST", ctx=mx.gpu(0))
else :
    print("Imported")


