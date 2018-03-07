import Network
import argparse

'''I wrote this code with reference to  https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py.
    I tried to make it easy to understand.
'''
'''
I initialized the Hyperparameters values introduced in 'DETAILS OF ADVERSORIAL TRAINING part'
of 'UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS' paper.
'''

'''Using argparser module, You can also run this code from the cmd window.'''

parser=argparse.ArgumentParser(description='<hyperparameters_setting>')
parser.add_argument("--state",action="store_true",help="state")
parser.add_argument("-e","--epoch",type=int, help="Total number of learning")
parser.add_argument("-n","--noise_size",type=int,help="Decide noise size!!!")
parser.add_argument("-b","--batch_size",type=int,help="Decide batch_size!!!")
parser.add_argument("-s","--save_period",type=int,help="Decide whether to store weights every few cycles")
parser.add_argument("-d","--dataset",type=str,help="select the dataset : MNIST? CIFAR10? ImageNet?")
args = parser.parse_args()

if args.state:
    print args.state
    Network.DCGAN(epoch=args.epoch, noise_size=args.noise_size, batch_size=args.batch_size, save_period=args.save_period,dataset=args.dataset)
else:
    #dataset : MNIST, CIFAR10 , Imagenet
    Network.DCGAN(epoch=300, noise_size=100, batch_size=128, save_period=100,dataset='CIFAR10')
