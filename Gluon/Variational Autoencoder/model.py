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
    return train_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    return train_data


#evaluate the data
def generate_image(network, n_latent, ctx , dataset):

    '''test'''
    column_size=10 ; row_size=10 #     column_size x row_size <= 10000
    output = network.decoder(nd.random.normal(loc=0, scale=1, shape=(column_size*row_size, n_latent),ctx=ctx))
    output = output.asnumpy() * 255.0

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

    elif dataset=="FashionMNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('FashionMNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Generate_Image/FashionMNIST_generator.png")

    plt.show()


class VAE_Network(gluon.HybridBlock):

    def __init__(self, n_hidden=300, n_latent=2, n_layers=1, n_output=784, batch_size=100, ctx=mx.gpu(0), act_type='relu', **kwargs):

        self.n_latent = n_latent
        self.batch_size = batch_size
        self.ctx=ctx

        super().__init__(**kwargs)
        with self.name_scope():
            #Encoder
            self.encoder = gluon.nn.HybridSequential(prefix='encoder')
            for i in range(n_layers):
                self.encoder.add(gluon.nn.Dense(n_hidden, activation=act_type))

            # mean, variance
            self.encoder.add(gluon.nn.Dense(n_latent * 2, activation=None))

            #Decoder
            self.decoder = gluon.nn.HybridSequential(prefix='decoder')
            for i in range(n_layers):
                self.decoder.add(gluon.nn.Dense(n_hidden, activation=act_type))
            self.decoder.add(gluon.nn.Dense(n_output, activation='sigmoid'))

    def hybrid_forward(self, F, x):

        #p(z|x) - it's difficult to calculate 'posterior distribution' - intractable.
        temp = self.encoder(x)

        mu_logvar = F.split(temp, axis=1, num_outputs=2)
        mu = mu_logvar[0]
        log_var = mu_logvar[1]

        #keypoint1
        # zero-mean Gaussians
        # reparametrization trick
        zero_mean_Gaussian = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=self.ctx)

        # e^(0.5*log(std)^2) = std
        #latent variable
        z = mu + F.exp(0.5 * log_var) * zero_mean_Gaussian

        #after learning completed, use Generative Model

        # p(x|z)
        y = self.decoder(z)
        # keypoint2+

        #reconstruction loss - cross entropy
        log_loss = F.sum(x * F.log(y + 1e-12) + (1 - x) * F.log(1 - y + 1e-12 ), axis=1)

        #KL Divergence Regularizer - Just accept it.
        KL = 0.5 * F.sum(F.exp(log_var) + mu * mu  -1 - log_var, axis=1)

        # keypoint3
        #ELBO = Evidence Lower Bound
        ELBO = log_loss - KL

        return -ELBO

def Variational_Autoencoder(epoch = 100 , batch_size=128, save_period=10 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset =="MNIST":
        train_data = MNIST(batch_size)
        path = "weights/MNIST-{}.params".format(load_period)
    elif dataset == "FashionMNIST":
        train_data = FashionMNIST(batch_size)
        path = "weights/FashionMNIST-{}.params".format(load_period)
    else:
        return "The dataset does not exist."

    #latent_dimension
    n_latent=16

    #Variational Autoencoder
    net = VAE_Network(n_hidden=100, n_latent=n_latent, n_layers=2, n_output=784, batch_size=batch_size, ctx=ctx) # stacks 'Block's sequentially
    net.hybridize()
    #weights initialization
    if os.path.exists(path):
        print("loading weights")
        net.load_params(filename=path , ctx=ctx) # weights load
    else:
        print("initializing weights")
        net.collect_params().initialize(mx.init.Normal(sigma=0.01),ctx=ctx) # weights initialization

    #optimizer
    trainer = gluon.Trainer(net.collect_params() , optimizer, {"learning_rate" : learning_rate})

    #learning
    for i in tqdm(range(1,epoch+1,1)):
        for data , label in train_data:

            data = data.as_in_context(ctx).reshape((batch_size,-1))

            with autograd.record(train_mode=True):
                loss=net(data)
                cost=nd.mean(loss).asscalar()
            loss.backward()
            trainer.step(batch_size,ignore_stale_grad=True)

        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="MNIST":
                net.save_params("weights/MNIST-{}.params".format(i))

            elif dataset=="FashionMNIST":
                net.save_params("weights/FashionMNIST-{}.params".format(i))

    #show image
    generate_image(net, n_latent, ctx , dataset)

    return "optimization completed"

if __name__ == "__main__":
    Variational_Autoencoder(epoch = 100 , batch_size=128, save_period=100 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0))
else :
    print("Imported")


