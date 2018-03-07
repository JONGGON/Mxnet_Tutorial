import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST
result=model.muitlclass_logistic_regression(epoch=20, batch_size=256 , save_period=20 , load_period=0 ,  weight_decay=0.001 , learning_rate=0.001, dataset="FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")