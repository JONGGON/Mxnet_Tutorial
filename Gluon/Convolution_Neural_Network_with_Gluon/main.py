import model
import mxnet as mx

#dataset = MNIST or CIFAR10 or FashionMNIST
result=model.CNN(epoch = 1, batch_size=128, save_period=100 , load_period=100 ,optimizer="adam",learning_rate= 0.001 , dataset = "FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")#implementation