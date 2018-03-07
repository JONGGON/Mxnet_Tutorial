import numpy as np
import os
import urllib
import gzip
import struct
import mxnet as mx

def download_data(url, force_download=True):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data_from_internet(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)

    '''
    <Label data should be one_hot encoded.>

    1.For using mxnet's one_hot encoding, the label data must be of type int32 unconditionally.
    thus , It should be written in the following format unconditionally.
    '''
    label_one_hot =  mx.nd.array(label,dtype=np.int32,ctx=mx.gpu(0))

    '''
    2. one hot encoding
    '''
    label_one_hot =  mx.nd.one_hot(label_one_hot , 10)

    return (label_one_hot,label, image)

def read_data_from_file(label, image):
    with gzip.open(label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)

    '''
    <Label data should be one_hot encoded.>

    1.For using mxnet's one_hot encoding, the label data must be of type int32 unconditionally.
    thus , It should be written in the following format unconditionally.
    '''
    label_one_hot =  mx.nd.array(label,dtype=np.int32,ctx=mx.gpu(0))

    '''
    2. one hot encoding
    '''
    label_one_hot =  mx.nd.one_hot(label_one_hot , 10)

    return (label_one_hot,label, image)


if __name__ == "__main__":

    '''print "download mnist data"
    path = 'http://yann.lecun.com/exdb/mnist/'
    (train_lbl_one_hot, train_lbl, train_img)  = read_data_from_internet(path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img)  = read_data_from_internet(path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')'''

    (train_lbl_one_hot, train_lbl, train_img) = read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = read_data_from_file('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz')

else:

    print "Load the mnist data"
