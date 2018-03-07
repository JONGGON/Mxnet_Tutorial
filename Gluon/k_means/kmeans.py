import random
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import *

class K_means(gluon.Block):

    def __init__(self,dataset,centroid,centroid_number,ctx,**kwargs):
        super(K_means, self).__init__(**kwargs)
        self.centroid_numbers=centroid_number
        self.dataset=dataset
        self.centroid=centroid
        self.ctx=ctx

    def forward(self):

        # 2-step
        diff = nd.subtract(nd.expand_dims(self.dataset,axis=0),nd.expand_dims(self.centroid,axis=1))
        sqr = nd.square(diff)
        distance = nd.sum(sqr,axis=2)
        clustering = nd.argmin(distance,axis=0)
        # 3-step
        '''
        Because mxnet's nd.where did not return the location. I wrote the np.where function.
        '''
        for j in range(self.centroid_numbers):
            self.centroid[j][:]=nd.mean(nd.take(self.dataset,nd.array(np.reshape(np.where(np.equal(clustering.asnumpy(), j)), (-1,)), ctx=self.ctx),axis=0),axis=0)
        return clustering , self.centroid

#I refer to the k-means code in the book, "TensorFlow First Steps."
def K_means_Algorithm(epoch=100,point_numbers=2000,centroid_numbers=5,ctx=mx.gpu(0)):

    dataset=[]
    centroid=[]

    # data generation
    for i in range(point_numbers):

        if random.random() > 0.5:
            dataset.append([np.random.normal(loc=0,scale=0.9),np.random.normal(loc=0,scale=0.9)])
        else:
            dataset.append([np.random.normal(loc=3,scale=0.5),np.random.normal(loc=0,scale=0.9)])

    df = pd.DataFrame({"x": [d[0] for d in dataset] , "y": [d[1] for d in dataset]})
    sns.lmplot("x","y" , data=df , fit_reg=False , size=10)
    plt.savefig("K means Algorithm init using mxnet gluon.png")

    # 1-step
    random.shuffle(dataset)
    for i in range(centroid_numbers):
        centroid.append(random.choice(dataset))

    # using mxnet
    dataset=nd.array(dataset, ctx=ctx)
    centroid=nd.array(centroid, ctx=ctx)

    net=K_means(dataset,centroid,centroid_numbers,ctx)
    # data assignment , updating new center values
    for i in tqdm(range(1,epoch,1)):
        print("epoch : {}".format(i+1))
        clustering, centroid=net()

    for i in range(centroid_numbers):
        print("{}_center : Final center_value : {}" . format(i+1,centroid.asnumpy()[i]))

    #4 show result
    data = {"x": [], "y": [] , "cluster" : []}
    for i in range(len(clustering)):
        data["x"].append(dataset[i][0].asscalar())
        data["y"].append(dataset[i][1].asscalar())
        data["cluster"].append(clustering[i].asscalar())

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=10 , hue="cluster")
    plt.savefig("K means Algorithm completed using mxnet gluon.png")
    plt.show()

if __name__ == "__main__":
    K_means_Algorithm(epoch=100, centroid_numbers=5, point_numbers=2000, ctx=mx.gpu(0))
else:
    print("mxnet kmeans Imported")
