import model
import mxnet as mx
#implementation

#dataset = FashionMNIST
result=model.CNN_Autoencoder(epoch=1, batch_size=128 , save_period=10, load_period=10 ,  weight_decay=0.0 , learning_rate=0.001, dataset="FashionMNIST", ctx=mx.gpu(0))
print("///"+result+"///")
