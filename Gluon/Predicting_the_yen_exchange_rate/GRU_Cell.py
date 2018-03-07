import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.autograd as autograd
import data_preprocessing as dp
from tqdm import *
import os

class GRUCell(gluon.rnn.HybridRecurrentCell):

    def __init__(self, hidden_size,output_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(GRUCell, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._input_size = input_size
        self.output_size=output_size
        self.i2h_weight = self.params.get('i2h_weight', shape=(3*hidden_size, input_size),
                                          init=i2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=(3*hidden_size, hidden_size),
                                          init=h2h_weight_initializer,
                                          allow_deferred_init=True)
        self.i2h_bias = self.params.get('i2h_bias', shape=(3*hidden_size,),
                                        init=i2h_bias_initializer,
                                        allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(3*hidden_size,),
                                        init=h2h_bias_initializer,
                                        allow_deferred_init=True)

        self.wo = self.params.get('output_weights', shape=(output_size,hidden_size))
        self.bo = self.params.get('output_bias', shape=(output_size,))


    def hybrid_forward(self, F, inputs, states, i2h_weight,
                       h2h_weight, i2h_bias, h2h_bias, wo , bo):
        # pylint: disable=too-many-locals
        prefix = 't%d_'%self._counter
        prev_state_h = states[0]
        i2h = F.FullyConnected(data=inputs,
                               weight=i2h_weight,
                               bias=i2h_bias,
                               num_hidden=self._hidden_size * 3,
                               name=prefix+'i2h')
        h2h = F.FullyConnected(data=prev_state_h,
                               weight=h2h_weight,
                               bias=h2h_bias,
                               num_hidden=self._hidden_size * 3,
                               name=prefix+'h2h')

        i2h_r, i2h_z, i2h = F.SliceChannel(i2h, num_outputs=3,
                                           name=prefix+'i2h_slice')
        h2h_r, h2h_z, h2h = F.SliceChannel(h2h, num_outputs=3,
                                           name=prefix+'h2h_slice')

        reset_gate = F.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                  name=prefix+'r_act')
        update_gate = F.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                   name=prefix+'z_act')

        next_h_tmp = F.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                  name=prefix+'h_act')

        next_h = F._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                   name=prefix+'out')

        output = F.FullyConnected(next_h , weight=wo,bias=bo, num_hidden=self.output_size)

        return output, [next_h]

def JPY_to_KRW(time_step,day,normalization_factor):
    training = gluon.data.DataLoader(dp.JPY_to_KRW(train=True,time_step=time_step, day=day,normalization_factor=normalization_factor), batch_size=time_step) #Loads data from a dataset and returns mini-batches of data.
    prediction = gluon.data.DataLoader(dp.JPY_to_KRW(train=False,time_step=time_step, day=day,normalization_factor=normalization_factor), batch_size=time_step)  # Loads data from a dataset and returns mini-batches of data.
    return training, prediction

def prediction(test_data, time_step, day, normalization_factor, num_hidden, model, ctx):

    for data, label in test_data:
        states = [nd.zeros(shape=(1, num_hidden), ctx=ctx)]
        data = data.as_in_context(ctx)
        data = data.reshape(shape=(-1, time_step, day))
        data = nd.transpose(data=data, axes=(1, 0, 2))

        outputs_list=[]
        for j in range(time_step):
            outputs, states = model(data[j], states)
            outputs_list.append(outputs.asnumpy())

    outputs_list = np.array(outputs_list) * normalization_factor
    outputs_list= np.reshape(outputs_list,(-1,))

    print("KRW-JPY exchange rate prediction for November 27th.")
    print("prediction value : {}".format(outputs_list[-1]))
    print("real value : {}".format(971.66))

def exchange_rate_model(epoch=1000, time_step=28, day=7, normalization_factor=100, save_period=1000 , load_period=1000 , learning_rate=0.001, ctx=mx.gpu(0)):

    ''' 28 time x 1 day '''
    #network parameter
    normalization_factor=normalization_factor
    time_step = time_step # 28  step
    day = day # 1 day
    num_hidden = 300

    training, test = JPY_to_KRW(time_step,day,normalization_factor)

    path = "weights/GRUCell_weights-{}.params".format(load_period)
    model=GRUCell(num_hidden,day)
    model.hybridize()

    # weight initialization
    if os.path.exists(path):
        print("loading weights")
        model.load_params(filename=path ,ctx=ctx) # weights load
    else:
        print("initializing weights")
        model.collect_params().initialize(mx.init.Normal(sigma=0.01),ctx=ctx) # weights initialization

    trainer = gluon.Trainer(model.collect_params(), "rmsprop", {"learning_rate": learning_rate})

    for i in tqdm(range(1,epoch+1,1)):
        for data,label in training:
            states = [nd.zeros(shape=(1, num_hidden), ctx=ctx)]
            data=data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            data = data.reshape(shape=(-1,time_step, day))
            data= nd.transpose(data=data,axes=(1,0,2))

            loss=0
            with autograd.record():
                for j in range(time_step):
                    outputs , states = model(data[j],states)
                    loss = loss + gluon.loss.L2Loss()(outputs,label[j].reshape(shape=outputs.shape))
            loss.backward()
            trainer.step(batch_size=1)
        cost = nd.mean(loss).asscalar()
        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            model.save_params("weights/GRUCell_weights-{}.params".format(i))

    prediction(test, time_step, day, normalization_factor ,num_hidden, model, ctx)

if __name__ == "__main__":
    exchange_rate_model(epoch=0, time_step=28, day=7, normalization_factor=100, save_period=1000 , load_period=1000 , learning_rate=0.001, ctx=mx.gpu(0))
else :
    print("GRU Cell Imported")


