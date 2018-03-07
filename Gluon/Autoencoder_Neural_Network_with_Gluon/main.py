import model
import mxnet as mx
#implementation

#dataset = MNIST or FashionMNIST
result=model.Autoencoder(epoch = 0, batch_size=128, save_period=100 , load_period=100 ,optimizer="adam",learning_rate= 0.001 , dataset = "FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")