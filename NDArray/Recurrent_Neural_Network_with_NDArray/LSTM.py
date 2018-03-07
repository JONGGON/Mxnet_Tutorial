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
def evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, LSTM_Cell, ctx):

    numerator = 0
    denominator = 0

    for data, label in test_data:
        h_state = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
        c_state = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
        data = data.as_in_context(ctx)
        data = data.reshape(shape=(-1, time_step, num_inputs))
        data = nd.transpose(data=data, axes=(1, 0, 2))
        label = label.as_in_context(ctx)

        outputs, h_state, c_state = LSTM_Cell(data, h_state, c_state)

        predictions = nd.argmax(outputs, axis=1) #(batch_size,)
        predictions = predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += predictions.shape[0]

    return (numerator / denominator)

def LSTM(epoch = 100 , batch_size=100, save_period=100 , load_period=100 ,learning_rate= 0.1, ctx=mx.gpu(0)):

    train_data , test_data = FashionMNIST(batch_size)

    #network parameter
    time_step = 28
    num_inputs = 28
    num_hidden = 200
    num_outputs = 10

    path = "weights/FashionMNIST_LSTMweights-{}".format(load_period)

    if os.path.exists(path):

        print("loading weights")
        [wxhf, wxhi, wxho, wxhg, whhf, whhi, whho, whhg, bhf, bhi, bho, bhg, why, by] = nd.load(path)  # weights load
        wxhf = wxhf.as_in_context(ctx)
        wxhi = wxhi.as_in_context(ctx)
        wxho = wxho.as_in_context(ctx)
        wxhg = wxhg.as_in_context(ctx)

        whhf = whhf.as_in_context(ctx)
        whhi = whhi.as_in_context(ctx)
        whho = whho.as_in_context(ctx)
        whhg = whhg.as_in_context(ctx)

        bhf = bhf.as_in_context(ctx)
        bhi = bhi.as_in_context(ctx)
        bho = bho.as_in_context(ctx)
        bhg = bhg.as_in_context(ctx)

        why = why.as_in_context(ctx)
        by = by.as_in_context(ctx)
        params = [wxhf , wxhi , wxho , wxhg, whhf, whhi, whho, whhg, bhf, bhi, bho, bhg, why , by]

    else:
        print("initializing weights")

        with ctx :
            wxhf = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))
            wxhi = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))
            wxho = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))
            wxhg = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_inputs))

            whhf = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))
            whhi = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))
            whho = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))
            whhg = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_hidden))

            bhf = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))
            bhi = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))
            bho = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))
            bhg = nd.random.normal(loc=0,scale=0.01,shape=(num_hidden,))

            why = nd.random.normal(loc=0,scale=0.1,shape=(num_outputs , num_hidden))
            by = nd.random.normal(loc=0,scale=0.1,shape=(num_outputs,))

        params = [wxhf , wxhi , wxho , wxhg, whhf, whhi, whho, whhg, bhf, bhi, bho, bhg, why , by]

    # attach gradient!!!
    for param in params:
        param.attach_grad()

    #Fully Neural Network with 1 Hidden layer
    def LSTM_Cell(input, h_state, c_state):
        for x in input:
            f_t = nd.Activation(nd.FullyConnected(data=x,weight=wxhf,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=h_state,weight=whhf,no_bias=True,num_hidden=num_hidden)+bhf,act_type="sigmoid")
            i_t = nd.Activation(nd.FullyConnected(data=x,weight=wxhi,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=h_state,weight=whhi,no_bias=True,num_hidden=num_hidden)+bhi,act_type="sigmoid")
            o_t = nd.Activation(nd.FullyConnected(data=x,weight=wxho,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=h_state,weight=whho,no_bias=True,num_hidden=num_hidden)+bho,act_type="sigmoid")
            g_t = nd.Activation(nd.FullyConnected(data=x,weight=wxhg,no_bias=True,num_hidden=num_hidden)+
                                nd.FullyConnected(data=h_state,weight=whhg,no_bias=True,num_hidden=num_hidden)+bhg,act_type="tanh")
            c_state = nd.multiply(f_t, c_state) + nd.multiply(i_t,g_t)
            h_state = nd.multiply(o_t,nd.tanh(c_state))

        output = nd.FullyConnected(data=h_state, weight=why, bias=by, num_hidden=num_outputs)
        output = nd.softmax(data=output)
        return output, h_state, c_state

    def cross_entropy(output, label):
        return - nd.sum(label * nd.log(output), axis=0 , exclude=True)

    #Adam optimizer
    state=[]
    optimizer=mx.optimizer.Adam(rescale_grad=1,learning_rate=learning_rate)

    for param in params:
        state.append(optimizer.create_state(0,param))

    for i in tqdm(range(1,epoch+1,1)):

        for data,label in train_data:

            h_state = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
            c_state = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)

            data = data.as_in_context(ctx)
            data = data.reshape(shape=(-1,time_step,num_inputs))
            data=nd.transpose(data=data,axes=(1,0,2))
            label = label.as_in_context(ctx)
            label = nd.one_hot(label , num_outputs)

            with autograd.record():
                outputs, h_state, c_state = LSTM_Cell(data, h_state , c_state)
                loss = cross_entropy(outputs,label) # (batch_size,)
            loss.backward()

            cost = nd.mean(loss).asscalar()
            for j,param in enumerate(params):
                optimizer.update(0,param,param.grad,state[j])

        test_accuracy = evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, LSTM_Cell, ctx)
        print(" epoch : {} , last batch cost : {}".format(i,cost))
        print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

        #weight_save
        if i % save_period==0:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            print("saving weights")
            nd.save("weights/FashionMNIST_LSTMweights-{}".format(i),params)

    test_accuracy = evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, LSTM_Cell, ctx)
    print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))
    return "optimization completed"

if __name__ == "__main__":
    LSTM(epoch=100, batch_size=128, save_period=100 , load_period=100 ,learning_rate=0.001, ctx=mx.gpu(0))
else :
    print("Imported")


