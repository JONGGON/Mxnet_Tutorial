import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.autograd as autograd
from tqdm import *
import os

class LSTMCell(gluon.rnn.HybridRecurrentCell):

    def __init__(self, hidden_size,output_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(LSTMCell, self).__init__(prefix=prefix, params=params)

        self._hidden_size = hidden_size
        self.output_size = output_size
        self._input_size = input_size
        self.i2h_weight = self.params.get('i2h_weight', shape=(4*hidden_size, input_size),
                                          dtype=None, init=i2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=(4*hidden_size, hidden_size),
                                          dtype=None, init=h2h_weight_initializer,
                                          allow_deferred_init=True)
        self.i2h_bias = self.params.get('i2h_bias', shape=(4*hidden_size,),
                                        dtype=None, init=i2h_bias_initializer,
                                        allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(4*hidden_size,),
                                        dtype=None, init=h2h_bias_initializer,
                                        allow_deferred_init=True)

        self.wo = self.params.get('output_weights', shape=(output_size,hidden_size))
        self.bo = self.params.get('output_bias', shape=(output_size,))

    def hybrid_forward(self, F, inputs, states, i2h_weight,
                       h2h_weight, i2h_bias, h2h_bias, wo, bo):
        prefix = 't%d_'%self._counter
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size*4, name=prefix+'i2h')
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias,
                               num_hidden=self._hidden_size*4, name=prefix+'h2h')
        gates = i2h + h2h
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix+'slice')
        in_gate = F.Activation(slice_gates[0], act_type="sigmoid", name=prefix+'i')
        forget_gate = F.Activation(slice_gates[1], act_type="sigmoid", name=prefix+'f')
        in_transform = F.Activation(slice_gates[2], act_type="tanh", name=prefix+'c')
        out_gate = F.Activation(slice_gates[3], act_type="sigmoid", name=prefix+'o')
        next_c = F._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                   name=prefix+'state')
        next_h = F._internal._mul(out_gate, F.Activation(next_c, act_type="tanh"),
                                  name=prefix+'out')

        output = F.FullyConnected(next_h,weight=wo,bias=bo, num_hidden=self.output_size)

        return output, [next_h, next_c]


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#MNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True ) #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST", train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.
    return train_data , test_data

#evaluate the data
def evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, model, ctx):

    numerator = 0
    denominator = 0
    for data, label in test_data:
        H_states = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
        C_states = nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
        data = data.as_in_context(ctx)
        data = data.reshape(shape=(-1, time_step, num_inputs))
        data = nd.transpose(data=data, axes=(1, 0, 2))
        label = label.as_in_context(ctx)

        for j in range(time_step):
            outputs, [H_states, C_states] = model(data[j], [H_states, C_states])  # outputs => (batch size, 10)

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

    path = "weights/FashionMNIST_LSTMweights-{}.params".format(load_period)
    model=LSTMCell(num_hidden,num_outputs)
    model.hybridize()

    # weight initialization
    if os.path.exists(path):
        print("loading weights")
        model.load_params(filename=path ,ctx=ctx) # weights load
    else:
        print("initializing weights")
        model.collect_params().initialize(mx.init.Normal(sigma=0.01),ctx=ctx) # weights initialization

    trainer = gluon.Trainer(model.collect_params(), "adam", {"learning_rate": learning_rate})
    for i in tqdm(range(1,epoch+1,1)):

        for data,label in train_data:
            H_states=nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
            C_states=nd.zeros(shape=(data.shape[0], num_hidden), ctx=ctx)
            data=data.as_in_context(ctx)
            data = data.reshape(shape=(-1,time_step,num_inputs))
            data= nd.transpose(data=data,axes=(1,0,2))
            label = label.as_in_context(ctx)

            with autograd.record():
                for j in range(time_step):
                    outputs , [H_states , C_states] = model(data[j],[H_states,C_states]) #outputs => (batch size, 10)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(outputs,label) # (batch_size,)

            loss.backward()
            trainer.step(batch_size)

        cost = nd.mean(loss).asscalar()
        test_accuracy = evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, model, ctx)
        print(" epoch : {} , last batch cost : {}".format(i,cost))
        print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            model.save_params("weights/FashionMNIST_LSTMweights-{}.params".format(i))

    test_accuracy = evaluate_accuracy(test_data, time_step, num_inputs, num_hidden, model, ctx)
    print("Test_acc : {0:0.3f}%".format(test_accuracy * 100))

if __name__ == "__main__":
    LSTM(epoch=100, batch_size=128, save_period=100 , load_period=100 ,learning_rate=0.001, ctx=mx.gpu(0))
else :
    print("Imported")


