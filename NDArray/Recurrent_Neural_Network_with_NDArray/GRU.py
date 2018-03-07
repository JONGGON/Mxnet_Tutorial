import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.autograd as autograd
from tqdm import *
import os

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#MNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True) #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST", train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.
    return train_data , test_data

#evaluate the data
def evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, GRU_Cell, ctx):

    numerator = 0
    denominator = 0

    for data, label in test_data:
        states = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
        data = data.as_in_context(ctx)
        data = data.reshape(shape=(-1, time_step, num_inputs))
        data = nd.transpose(data=data, axes=(1, 0, 2))
        label = label.as_in_context(ctx)

        outputs , states = GRU_Cell(data,states)

        predictions = nd.argmax(outputs, axis=1) #(batch_size,)
        predictions = predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += predictions.shape[0]

    return (numerator / denominator)

def GRU(epoch = 100 , batch_size=100, save_period=100 , load_period=100 ,learning_rate= 0.1, ctx=mx.gpu(0)):

    train_data , test_data = FashionMNIST(batch_size)

    #network parameter
    time_step = 28
    num_inputs = 28
    num_hidden = 200
    num_outputs = 10

    path = "weights/FashionMNIST_GRUweights-{}".format(load_period)

    if os.path.exists(path):

        print("loading weights")
        [wxz, wxr, wxh, whz, whr, whh, bz, br, bh, why, by] = nd.load(path)  # weights load
        wxz = wxz.as_in_context(ctx)
        wxr = wxr.as_in_context(ctx)
        whz = whz.as_in_context(ctx)


        whz = whz.as_in_context(ctx)
        whr = whr.as_in_context(ctx)
        whh = whh.as_in_context(ctx)

        bz = bz.as_in_context(ctx)
        br = br.as_in_context(ctx)
        bh = bh.as_in_context(ctx)

        why = why.as_in_context(ctx)
        by = by.as_in_context(ctx)
        params = [wxz , wxr , wxh , whz, whr, whh, bz, br, bh, why , by]

    else:
        print("initializing weights")

        with ctx:
            wxz = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))
            wxr = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))
            wxh = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))

            whz = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))
            whr = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))
            whh = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))

            bz = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))
            br = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))
            bh = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))

            why = nd.random.normal(loc=0,scale=0.1,shape=(num_outputs , num_hidden))
            by = nd.random.normal(loc=0,scale=0.1,shape=(num_outputs,))

        params = [wxz , wxr , wxh , whz, whr, whh, bz, br, bh, why , by]

    # attach gradient!!!
    for param in params:
        param.attach_grad()

    #Fully Neural Network with 1 Hidden layer
    def GRU_Cell(input, state):
        for x in input:
            z_t = nd.Activation(nd.FullyConnected(data=x,weight=wxz,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=state,weight=whz,no_bias=True,num_hidden=num_hidden)+bz,act_type="sigmoid")
            r_t = nd.Activation(nd.FullyConnected(data=x,weight=wxr,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=state,weight=whr,no_bias=True,num_hidden=num_hidden)+br,act_type="sigmoid")
            g_t = nd.Activation(nd.FullyConnected(data=x,weight=wxh,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=r_t*state,weight=whh,no_bias=True,num_hidden=num_hidden)+bh,act_type="tanh")

            state = nd.multiply(z_t,state) + nd.multiply(1-z_t,g_t)

        output = nd.FullyConnected(data=state, weight=why, bias=by, num_hidden=num_outputs)
        output = nd.softmax(data=output)
        return output, state

    def cross_entropy(output, label):
        return - nd.sum(label * nd.log(output), axis=0 , exclude=True)

    #Adam optimizer
    state=[]
    optimizer=mx.optimizer.Adam(rescale_grad=1,learning_rate=learning_rate)

    for param in params:
        state.append(optimizer.create_state(0,param))

    for i in tqdm(range(1,epoch+1,1)):

        for data,label in train_data:

            states = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
            data = data.as_in_context(ctx)
            data = data.reshape(shape=(-1,time_step,num_inputs))
            data=nd.transpose(data=data,axes=(1,0,2))
            label = label.as_in_context(ctx)
            label = nd.one_hot(label , num_outputs)

            with autograd.record():
                outputs, states = GRU_Cell(data, states)
                loss = cross_entropy(outputs,label) # (batch_size,)
            loss.backward()

            cost = nd.mean(loss).asscalar()
            for j,param in enumerate(params):
                optimizer.update(0,param,param.grad,state[j])

        test_accuracy = evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, GRU_Cell, ctx)
        print(" epoch : {} , last batch cost : {}".format(i,cost))
        print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

        #weight_save
        if i % save_period==0:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            print("saving weights")
            nd.save("weights/FashionMNIST_GRUweights-{}".format(i),params)

    test_accuracy = evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, GRU_Cell, ctx)
    print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))
    return "optimization completed"

if __name__ == "__main__":
    GRU(epoch=100, batch_size=128, save_period=100 , load_period=100 ,learning_rate=0.001, ctx=mx.gpu(0))
else :
    print("Imported")


