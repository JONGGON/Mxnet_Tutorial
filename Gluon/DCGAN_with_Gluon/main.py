import model
import mxnet as mx
#implementation

#dataset = CIFAR10 or FashionMNIST
result=model.DCGAN(epoch = 1, batch_size=128, save_period=100, load_period=100, optimizer="adam", beta1=0.5, learning_rate= 0.0002 , dataset = "FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")