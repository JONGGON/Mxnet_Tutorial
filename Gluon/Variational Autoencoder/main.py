import model
import mxnet as mx
#implementation

#dataset = MNIST or FashionMNIST
result=model.Variational_Autoencoder(epoch = 100, batch_size=256, save_period=100 , load_period=100 ,optimizer="adam",learning_rate= 0.001 , dataset = "MNIST", ctx=mx.gpu(0))
print("///"+result+"///")