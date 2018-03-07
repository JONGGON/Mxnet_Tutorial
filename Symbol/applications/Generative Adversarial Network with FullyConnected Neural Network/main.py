import Network

'''I wrote this code with reference to  https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py.
    I tried to make it easy to understand.
'''
Network.GAN(epoch=1, noise_size=100, batch_size=128,save_period=50,load_weights=100)