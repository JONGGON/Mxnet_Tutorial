import random
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.symbol as sym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import *

def init_kmeans(dataset):

    df = pd.DataFrame({"x": [d[0] for d in dataset] , "y": [d[1] for d in dataset]})
    sns.lmplot("x","y" , data=df , fit_reg=False , size=10)
    plt.savefig("K means Algorithm init using mxnet symbol.png")

def completed_kmeans(dataset, clustering):

    #4 show result
    data = {"x": [], "y": [] , "cluster" : []}
    for i in range(len(clustering)):
        data["x"].append(dataset[i][0].asscalar())
        data["y"].append(dataset[i][1].asscalar())
        data["cluster"].append(clustering[i].asscalar())

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=10 , hue="cluster")
    plt.savefig("K means Algorithm completed using mxnet symbol.png")
    plt.show()

def K_means_assignment():

    data = mx.sym.Variable('data')
    centroid = mx.sym.Variable("centroid")
    #data assignment
    with mx.name.Prefix("k_"):
        # 2-step
        diff = sym.broadcast_sub(sym.expand_dims(data, axis=0), sym.expand_dims(centroid, axis=1))
        sqr = sym.square(diff)
        distance = sym.sum(sqr, axis=2)
        clustering = sym.argmin(distance, axis=0)
    return clustering

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

    init_kmeans(dataset)

    # 1-step
    random.shuffle(dataset)
    for i in range(centroid_numbers):
        centroid.append(random.choice(dataset))

    # using mxnet
    dataset=nd.array(dataset, ctx=ctx)
    centroid=nd.array(centroid, ctx=ctx)

    # 2-step
    assignment = K_means_assignment()

    arg_names = assignment.list_arguments()
    arg_shapes, output_shapes, aux_shapes = assignment.infer_shape(data=(point_numbers,2) , centroid=(centroid_numbers,2))

    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape=shape, ctx=ctx) for shape in arg_shapes]))
    arg_dict['data'] = dataset
    arg_dict['centroid'] = centroid

    shape = {"data": (point_numbers,2), "centroid" : (centroid_numbers,2)}
    graph = mx.viz.plot_network(symbol=assignment, shape=shape)
    if epoch==1:
        graph.view("Kmeans_assignment")

    binder = assignment.bind(ctx=ctx, args=arg_dict)

    #updating new center values
    for i in tqdm(range(epoch)):
        print("epoch : {}".format(i+1))
        arg_dict['data'][:] = dataset
        clustering=binder.forward()

        # 3-step
        for j in range(centroid_numbers):
            centroid[j][:]=nd.mean(nd.take(dataset,nd.array(np.reshape(np.where(np.equal(clustering[0].asnumpy(), j)), (-1,)), ctx=ctx),axis=0),axis=0)

        arg_dict['centroid'][:] = centroid

    for i in range(centroid_numbers):
        print("{}_center : Final center_value : {}" . format(i+1,centroid.asnumpy()[i]))

    # show result
    completed_kmeans(dataset, clustering[0])

if __name__ == "__main__":
    K_means_Algorithm(epoch=100, centroid_numbers=5, point_numbers=2000, ctx=mx.gpu(0))
else:
    print("mxnet kmeans Imported")
