import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import *
import os

'''Deep Convolution Generative Adversarial Networks'''
'''
    The formula is as follows.
    The Deconvolution formula is output_size = stride(input_size-1)+kernel-2*pad
'''
def Noise(batch_size=None,ctx=None):
    return nd.random_uniform(low=-1, high=1, shape=(batch_size, 100, 1, 1), ctx=ctx)

class Generator(gluon.HybridBlock):
    def __init__(self , **kwargs):
        super(Generator , self).__init__(**kwargs)
        with self.name_scope():
            self.Deconv1=gluon.nn.Conv2DTranspose(channels=512, kernel_size=(4,4), strides=(1,1), padding=(0,0), use_bias=False, activation=None)
            self.BatchNorm1=gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True) #()
            self.Deconv2=gluon.nn.Conv2DTranspose(channels=256, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.BatchNorm2 = gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True)
            self.Deconv3=gluon.nn.Conv2DTranspose(channels=128, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.BatchNorm3 = gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True)
            self.Deconv4=gluon.nn.Conv2DTranspose(channels=64,kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.BatchNorm4 = gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True)
            self.Deconv5=gluon.nn.Conv2DTranspose(channels=3,kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation='tanh') # activation : tanh

    def hybrid_forward(self , F, x):
        x = F.relu(self.BatchNorm1(self.Deconv1(x))) #(batch_size , 512 , 4, 4)
        x = F.relu(self.BatchNorm2(self.Deconv2(x))) #(batch_size , 256 , 8 , 8)
        x = F.relu(self.BatchNorm3(self.Deconv3(x))) #(batch_size , 128 , 16 , 16)
        x = F.relu(self.BatchNorm4(self.Deconv4(x))) #(batch_size , 64 , 32 , 32)
        x = self.Deconv5(x)  # (batch_size , 3 , 64 , 64) # not applying batchnorm to the generator output : referenced by paper
        return x

'''
    The formula is as follows.

    The convolution formula is  output_size = ((input_size+2*pad-kernel_size) / stride) + 1
'''
class Discriminator(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1=gluon.nn.Conv2D(channels=64, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.conv2=gluon.nn.Conv2D(channels=128, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.BatchNorm2 = gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True)
            self.conv3=gluon.nn.Conv2D(channels=256, kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.BatchNorm3 = gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True)
            self.conv4=gluon.nn.Conv2D(channels=512,kernel_size=(4,4), strides=(2,2), padding=(1,1), use_bias=False, activation=None)
            self.BatchNorm4 = gluon.nn.BatchNorm(axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True)
            self.conv5=gluon.nn.Conv2D(channels=1,kernel_size=(4,4), strides=(1,1), padding=(0,0), use_bias=False, activation=None)

    def hybrid_forward(self , F, x):
        x = F.LeakyReLU(self.conv1(x), slope=0.2) #(batch_size , 64 , 32, 32)
        x = F.LeakyReLU(self.BatchNorm2(self.conv2(x)), slope=0.2) #(batch_size , 128 , 16 , 16)
        x = F.LeakyReLU(self.BatchNorm3(self.conv3(x)), slope=0.2) #(batch_size , 256 , 8 , 8)
        x = F.LeakyReLU(self.BatchNorm4(self.conv4(x)), slope=0.2) #(batch_size , 512 , 4 , 4)
        x = self.conv5(x) #(batch_size , 1 , 1 , 1)
        x = F.Flatten(x) #(batch_size,1)
        return x

def transform(data, label):

    data=data.asnumpy()
    data=cv2.resize(src=data, dsize=(64,64), interpolation=cv2.INTER_CUBIC)

    data=nd.array(data)
    if len(data.shape) == 2:
        data = data.reshape((64,64,1))
    data=nd.transpose(data.astype(np.float32), (2, 0, 1))
    data = data / 127.5 - 1
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return  data , label.astype(np.float32)

#CIFAR10 dataset
def CIFAR10(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label)
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = True, transform=transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = False, transform=transform) , 128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#evaluate the data
def generate_image(generator , ctx , dataset):

    # column_size x row_size
    column_size=10
    row_size=10

    generated_image=generator(Noise(batch_size=column_size*row_size, ctx=ctx))
    generated_image = ((generated_image+1)*127.5).astype("uint8")
    generated_image = [[cv2.resize(i, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for i in image] for image in generated_image.asnumpy()]
    generated_image = np.transpose(generated_image,axes=(0,2,3,1))

    print("show image")
    '''generator image visualization'''

    if not os.path.exists("Generate_Image"):
        os.makedirs("Generate_Image")

    fig, ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    if dataset=="FashionMNIST":
        fig.suptitle(dataset+'_generator')
    elif dataset=="CIFAR10":
        fig.suptitle(dataset+'_generator')
    for j in range(row_size):
        for i in range(column_size):
            ax[j][i].set_axis_off()
            if dataset == "FashionMNIST":
                ax[j][i].imshow(generated_image[i + j * column_size],cmap='gray')
            elif dataset == "CIFAR10":
                ax[j][i].imshow(generated_image[i + j * column_size])

    if dataset=="FashionMNIST":
        fig.savefig("Generate_Image/"+dataset+"_generator.png")
    elif dataset=="CIFAR10":
        fig.savefig("Generate_Image/"+dataset+"_generator.png")

    plt.show()

def DCGAN(epoch = 100, batch_size=128, save_period=10, load_period=100, optimizer="adam", beta1=0.5, learning_rate= 0.0002 , dataset = "FashionMNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset == "CIFAR10":
        train_data, test_data = CIFAR10(batch_size)
        G_path = "weights/CIFAR10-G{}.params".format(load_period)
        D_path = "weights/CIFAR10-D{}.params".format(load_period)
    elif dataset == "FashionMNIST":
        train_data, test_data = FashionMNIST(batch_size)
        G_path = "weights/FashionMNIST-G{}.params".format(load_period)
        D_path = "weights/FashionMNIST-D{}.params".format(load_period)
    else:
        return "The dataset does not exist."

    #network
    generator = Generator()
    discriminator = Discriminator()

    #for faster learning
    generator.hybridize()
    discriminator.hybridize()

    if os.path.exists(D_path) and os.path.exists(G_path):
        print("loading weights")
        generator.load_params(filename=G_path , ctx=ctx) # weights load
        discriminator.load_params(filename=D_path, ctx=ctx)  # weights load
    else:
        print("initializing weights")
        generator.collect_params().initialize(mx.init.Normal(sigma=0.02),ctx=ctx) # weights initialization
        discriminator.collect_params().initialize(mx.init.Normal(sigma=0.02), ctx=ctx)  # weights initialization
        #net.initialize(mx.init.Normal(sigma=0.1),ctx=ctx) # weights initialization

    #optimizer
    G_trainer = gluon.Trainer(generator.collect_params() , optimizer, {"learning_rate" : learning_rate , "beta1" : beta1})
    D_trainer = gluon.Trainer(discriminator.collect_params(), optimizer, {"learning_rate" : learning_rate , "beta1" : beta1})

    '''The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression.

    .. math::
        loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))


    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and then BCE, which is more numerically stable through
        log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch. '''

    SBCE= gluon.loss.SigmoidBCELoss()

    #learning
    start_time = time.time()

    #cost selection
    real_label = nd.ones((batch_size,),ctx=ctx)
    fake_label = nd.zeros((batch_size,),ctx=ctx)

    for i in tqdm(range(1,epoch+1,1)):
        for data , label in train_data:
            print("\n<<D(X) , G(X)>")
            data = data.as_in_context(ctx)
            noise = Noise(batch_size=batch_size, ctx=ctx)

            #1. Discriminator : (1)maximize Log(D(x)) + (2)Log(1-D(G(z)))
            with autograd.record(train_mode=True):
                output=discriminator(data)
                print("real_D(X) : {}".format(nd.mean(nd.sigmoid(output)).asscalar())),
                #(1)
                real=SBCE(output,real_label)
                #(2)
                fake_real=generator(noise)
                output=discriminator(fake_real)
                print("fake_real_D(X) : {}".format(nd.mean(nd.sigmoid(output)).asscalar()))
                fake_real=SBCE(output,fake_label)
                # cost definition
                discriminator_cost=real+fake_real

            discriminator_cost.backward()
            D_trainer.step(batch_size,ignore_stale_grad=True)

            # 2. Generator : (3)maximize Log(D(G(z)))
            with autograd.record(train_mode=True):

                fake=generator(noise)
                output=discriminator(fake)
                print("fake_G(X) : {}".format(nd.mean(nd.sigmoid(output)).asscalar()))

                #(3)
                Generator_cost=SBCE(output,real_label)

            Generator_cost.backward()
            G_trainer.step(batch_size,ignore_stale_grad=True)

        print(" epoch : {}".format(i))
        print("last batch Discriminator cost : {}".format(nd.mean(discriminator_cost).asscalar()))
        print("last batch Generator cost : {}".format(nd.mean(Generator_cost).asscalar()))

        if i % save_period==0:
            end_time = time.time()
            print("-------------------------------------------------------")
            print("{}_learning time : {}".format(epoch, end_time - start_time))
            print("-------------------------------------------------------")

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="FashionMNIST":
                generator.save_params("weights/FashionMNIST-G{}.params".format(i))
                discriminator.save_params("weights/FashionMNIST-D{}.params".format(i))
            elif dataset=="CIFAR10":
                generator.save_params("weights/CIFAR10-G{}.params".format(i))
                discriminator.save_params("weights/CIFAR10-D{}.params".format(i))

    #generate image
    generate_image(generator, ctx, dataset)
    return "optimization completed"


if __name__ == "__main__":
    DCGAN(epoch=1, batch_size=128, save_period=100, load_period=100, optimizer="adam", beta1=0.5, learning_rate=0.0002,dataset="FashionMNIST", ctx=mx.gpu(0))
else :
    print("Imported")


