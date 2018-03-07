import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import LottoData as lottodata
from tqdm import *
import os

#LOTTO dataset
def LOTTO(batch_size):
    train_data = gluon.data.DataLoader(lottodata.LOTTO(train=True) , batch_size=batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(lottodata.LOTTO(train=False) , batch_size=1 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.
    return train_data , test_data

def Prediction(test_data , network , ctx ):

    #Next Lotto Number!!!
    for data ,label in test_data :
        data = data.as_in_context(ctx)
        output = mx.nd.round(network(data))
        output = nd.clip(data=output, a_min=0, a_max=45)
        print(output.asnumpy()[0])

def LottoNet(epoch = 100 , batch_size=128, save_period=10 , load_period=100 ,optimizer="sgd",learning_rate= 0.001 , ctx=mx.gpu(0)):

    #Data Loading
    train_data, test_data = LOTTO(batch_size)
    path = "weights/lotto-{}.params".format(load_period)
    '''Follow these steps:

    •Define network
    •Initialize parameters
    •Loop over inputs
    •Forward input through network to get output
    •Compute loss with output and label
    •Backprop gradient
    •Update parameters with gradient descent.
    '''
    #Autoencoder
    network = gluon.nn.Sequential()  # stacks Hybrid'Block's sequentially for faster learning (using symbolic)
    #network = gluon.nn.HybridSequential() # stacks Hybrid'Block's sequentially for faster learning (using symbolic)

    with network.name_scope():

        network.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        network.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        network.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        network.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        network.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        network.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        network.add(gluon.nn.Dense(units=6 ,use_bias=True))

    #network.hybridize() # for faster
    #weights initialization
    if os.path.exists(path):
        print("loading weights")
        network.load_params(filename=path , ctx=ctx) # weights load
    else:
        print("initializing weights")
        network.collect_params().initialize(mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),ctx=ctx) # weights initialization
        #npnet.initialize(mx.init.Normal(sigma=0.1),ctx=ctx) # weights initialization

    #optimizer
    trainer = gluon.Trainer(network.collect_params() , optimizer, {"learning_rate" : learning_rate})

    #learning
    for i in tqdm(range(1,epoch+1,1)):
        for data , label in train_data:

            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record(train_mode=True):

                output=network(data)
                loss=gluon.loss.L2Loss()(output,label)
                cost=nd.mean(loss).asscalar()

            loss.backward()
            trainer.step(batch_size,ignore_stale_grad=True)

        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            network.save_params("weights/lotto-{}.params".format(i))

    #Predict Lotto Number
    Prediction(test_data , network , ctx )

    return "optimization completed"

if __name__ == "__main__":
    LottoNet(epoch = 30000 , batch_size=50, save_period=30000 , load_period=30000 ,optimizer="sgd",learning_rate= 0.01 , ctx=mx.gpu(0))
else :
    print("Imported")


