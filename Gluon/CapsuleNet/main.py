import model
import mxnet as mx
'''
Implementing 'Dynamic Routing Betwenn Capsules' Using Gluon
Note: GPU memory usage is 5 GB.
'''
#dataset = MNIST or FashionMNIST
result=model.CapsuleNet(Reconstruction=True, epoch = 0 , batch_size=128, save_period = 5, load_period = 5, optimizer="adam", learning_rate= 0.001, dataset = "MNIST", ctx=mx.gpu(0))
print("///"+result+"///")