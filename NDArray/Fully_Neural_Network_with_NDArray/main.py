import model
import mxnet as mx
#implementation

#dataset = MNIST or CIFAR10 or FashionMNIST
result=model.FNN(epoch=1, batch_size=256 , save_period=10 , load_period=10 ,  weight_decay=0.0001 , learning_rate=0.001, dataset="FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")