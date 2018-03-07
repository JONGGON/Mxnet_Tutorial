# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
import os
logging.basicConfig(level=logging.INFO)

'''Let's make my own layer in symbol.'''



#If you want to know more, go to mx.operator.CustomOp.
class Activation(mx.operator.CustomOp):
    '''
    If you want fast speed
    Proceed to mx.ndarray.function !!!
    '''
    def __init__(self,act_type):
        self.act_type=act_type

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data[0] -> "input" shape -> (batch_size , num_hidden)
        out_data[0] -> "output" shape -> (batch_size , num_hidden)
        '''
        #method1
        '''
        It is not very good to construct an 'if statement' as shown below. It is good to just make each.
        '''
        if self.act_type == "relu":
            Activation = mx.nd.maximum(in_data[0],0)

        elif self.act_type =="sigmoid":
            Activation = mx.nd.divide(1,1+mx.nd.exp(-in_data[0]))

        elif self.act_type == "tanh":
            Activation = mx.nd.divide(mx.nd.exp(in_data[0])-mx.nd.exp(-in_data[0]),mx.nd.exp(in_data[0])+mx.nd.exp(-in_data[0]))

        out_data[0][:]=Activation
        #method2
        """Helper function for assigning into dst depending on requirements."""
        #if necessary
        #self.assign(out_data[0], req[0], Activation)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        '''
        in_data[0] -> "input" shape -> (batch_size , num_hidden)
        out_data[0] -> "output" shape -> (batch_size , num_hidden)
        '''
        #method1
        if self.act_type == "relu":
            diff=mx.nd.where(condition=(in_data[0]>0) , x=mx.nd.ones(in_data[0].shape), y=mx.nd.zeros(in_data[0].shape))

        elif self.act_type =="sigmoid":
            sigmoid = mx.nd.divide(1,1+mx.nd.exp(-in_data[0]))
            diff = mx.nd.multiply(sigmoid,1-sigmoid)

        elif self.act_type == "tanh":
            tanh = mx.nd.divide(mx.nd.exp(in_data[0])-mx.nd.exp(-in_data[0]),mx.nd.exp(in_data[0])+mx.nd.exp(-in_data[0]))
            diff = mx.nd.subtract(1,mx.nd.square(tanh))

        in_grad[0][:]= mx.nd.multiply(out_grad[0],diff)
        #method2
        '''Helper function for assigning into dst depending on requirements.'''
        #if necessary
        #self.assign(in_grad[0], req[0], mx.nd.multiply(out_grad[0],diff))


#If you want to know more, go to mx.operator.CustomOpProp.
@mx.operator.register("Activation")
class ActivationProp(mx.operator.CustomOpProp):

    def __init__(self,act_type):
        '''
            need_top_grad : bool
        The default declare_backward_dependency function. Use this value
        to determine whether this operator needs gradient input.
        '''
        self.act_type=act_type
        super(ActivationProp, self).__init__(need_top_grad=True) # Must be true !!!

    #Required.
    def list_arguments(self):
        return ['data'] # Be sure to write this down. It is a keyword.

    #Required.
    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    #Can be omitted
    def list_outputs(self):
        return ['output']

    #Can be omitted
    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return Activation(self.act_type)

class SoftmaxOutput(mx.operator.CustomOp):

    '''
    If you want fast speed
    Proceed to mx.ndarray.function !!!
    '''

    def __init__(self, grad_scale):
        #grad_scale -> str
        self.grad_scale = float(grad_scale) #You need to change type to int or float.

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data[0] -> "input" shape -> (batch size , the number of class)
        in_data[1] -> "label" shape -> (batch size , the number of class)
        out_data[0] -> "output" shape -> (batch size , the number of class)
        '''
        numerator = mx.nd.exp(in_data[0]-mx.nd.max(in_data[0])) # normalization
        denominator = mx.nd.nansum(numerator, axis=0 , keepdims=True , exclude=True)

        #method1
        out_data[0][:]= mx.nd.divide(numerator,denominator)

        #method2
        """Helper function for assigning into dst depending on requirements."""
        #if necessary
        #self.assign(out_data[0], req[0], mx.nd.divide(numerator,denominator))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        '''
        in_data[0] -> "input" shape -> (batch size , the number of class)
        in_data[1] -> "label" shape -> (batch size , the number of class)
        out_data[0] -> "output" shape -> (batch size , the number of class)
        '''
        #method1
        #CrossEntropy
        in_grad[0][:] = (out_data[0] - in_data[1])*self.grad_scale

        #method2
        """Helper function for assigning into dst depending on requirements."""
        #if necessary
        #self.assign(in_grad[0], req[0], (out_data[0] - in_data[1])*self.grad_scale)

#If you want to know more, go to mx.operator.CustomOpProp.
@mx.operator.register("SoftmaxOutput")
class SoftmaxOutputProp(mx.operator.CustomOpProp):

    def __init__(self,grad_scale):

        self.grad_scale=grad_scale
        '''
            need_top_grad : bool
        The default declare_backward_dependency function. Use this value
        to determine whether this operator needs gradient input.
        '''
        super(SoftmaxOutputProp, self).__init__(False)

    #Required.
    def list_arguments(self):
        return ['data', 'label'] # Be sure to write this down. It is a keyword.

    #Required.
    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[0]], [in_shape[0]] , []

    #Can be omitted
    def list_outputs(self):
        return ['output']

    #Can be omitted
    def infer_type(self, in_type):
        return in_type, [in_type[0]], []
    '''
    #Define a create_operator function that will be called by the back-end 
    to create an instance of softmax:
    '''
    def create_operator(self, ctx, shapes, dtypes):
        return SoftmaxOutput(self.grad_scale)

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

def NeuralNet(epoch,batch_size,save_period,load_weights):

    '''
    load_data

    1. SoftmaxOutput must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data

    2. LogisticRegressionOutput , LinearRegressionOutput , MakeLoss and so on.. must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    '''

    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'one_hot_label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to2d(test_img)}, label={'one_hot_label' : test_lbl_one_hot}) #test data

    '''neural network'''
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('one_hot_label')

    # first_hidden_layer
    affine1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=100)
    hidden1 = mx.sym.Custom(data=affine1 ,act_type="tanh", op_type = "Activation") # sigmoid , tanh , relu available

    # two_hidden_layer
    affine2 = mx.sym.FullyConnected(data=hidden1, name='fc2', num_hidden=100)
    hidden2 = mx.sym.Custom(data=affine2 ,act_type="tanh", op_type = "Activation") # sigmoid , tanh , relu available

    # output_layer
    output_affine = mx.sym.FullyConnected(data=hidden2, name='fc3', num_hidden=10)
    output = mx.sym.Custom(data= output_affine , label = label , grad_scale = 1 , name="Softmax", op_type = 'SoftmaxOutput') #

    # We visualize the network structure with output size (the batch_size is ignored.)
    shape = {"data": (batch_size,784)}
    graph=mx.viz.plot_network(symbol=output,shape=shape)#The diagram can be found on the Jupiter notebook.
    if epoch==1:
        graph.view()

    # training mod
    mod = mx.mod.Module(symbol=output, data_names=['data'], label_names=['one_hot_label'], context=mx.gpu(0))
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    #load the saved mod data
    weghts_path="weights/mod-{}.params".format(load_weights)

    if os.path.exists(weghts_path) :
        print("Load weights")
        mod.load_params(weghts_path)
    else :
        mod.init_params(initializer=mx.initializer.Normal(sigma=0.01))

    mod.init_optimizer(optimizer='sgd',optimizer_params={'learning_rate': 0.1})

    # test mod
    test = mx.mod.Module(symbol=output, data_names=['data'], label_names=['one_hot_label'], context=mx.gpu(0))

    '''load method1 - using the shared_module'''
    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    """
    test.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label,shared_module=mod,for_training=False)

    # Network information print
    print(output.list_arguments())
    print(output.list_outputs())
    print(mod.data_names)
    print(mod.label_names)
    print(train_iter.provide_data)
    print(train_iter.provide_label)

    '''############Although not required, the following code should be declared.#################'''

    '''make evaluation method 1 - Using existing ones.
        metrics = {
        'acc': Accuracy,
        'accuracy': Accuracy,
        'ce': CrossEntropy,
        'f1': F1,
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'top_k_accuracy': TopKAccuracy
    }'''

    metric = mx.metric.create(['acc','mse'])

    '''make evaluation method 2 - Making new things.'''
    '''
    Custom evaluation metric that takes a NDArray function.
    Parameters:
    •feval (callable(label, pred)) – Customized evaluation function.
    •name (str, optional) – The name of the metric.
    •allow_extra_outputs (bool) – If true, the prediction outputs can have extra outputs.
    This is useful in RNN, where the states are also produced in outputs for forwarding.
    '''

    def zero(label, pred):
        return 0

    null = mx.metric.CustomMetric(zero)

    for epoch in range(1,epoch+1,1):
        print("epoch : {}".format(epoch))
        train_iter.reset()
        #total_batch_number = np.ceil(len(train_img) / (batch_size * 1.0))
        #temp=0
        for batch in train_iter:
            mod.forward(batch, is_train=True)
            mod.backward()
            mod.update()

            #cost
            #temp+=(mod.get_outputs()[0].asnumpy()-batch.label[0].asnumpy())

        #cost = (0.5*np.square(temp)/(total_batch_number*1.0)).mean()
        result = test.predict(test_iter).asnumpy().argmax(axis=1)
        print("training_data : {}".format(mod.score(train_iter, ['mse', 'acc'])))
        print('accuracy during learning.  : {}%'.format(float(sum(test_lbl == result)) / len(result) * 100.0))
        #print("cost value : {}".format(cost))

        if not os.path.exists("weights"):
            os.makedirs("weights")

        #Save the data
        if epoch%save_period==0:
            print('Saving weights')
            mod.save_params("weights/mod-{}.params" .format(epoch))

    # Network information print
    print(mod.data_shapes)
    print(mod.label_shapes)
    print(mod.output_shapes)
    print(mod.get_params())
    print(mod.get_outputs())

    print("Optimization complete.")
    #################################TEST####################################
    '''load method2 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''Annotate only when running test data. and Uncomment only if it is 'load method2' '''
    #test.set_params(arg_params, aux_params)

    #batch by batch accuracy
    #To use the code below, Test / batchsize must be an integer.
    '''for preds, i_batch, eval_batch in mod.iter_predict(test_iter):
        pred_label = preds[0].asnumpy().argmax(axis=1)
        label = eval_batch.label[0].asnumpy().argmax(axis=1)
        print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label == label)) / len(label)))
    '''
    '''test'''
    result = test.predict(test_iter).asnumpy().argmax(axis=1)
    print('Final accuracy : {}%' .format(float(sum(test_lbl == result)) / len(result)*100.0))

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100)
else:
    print("NeuralNet_imported")

