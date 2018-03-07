import cPickle
import numpy as np
import matplotlib.pyplot as plt


'''cifar10 data'''

path="CIFAR10/"
row_size = 10
column_size = 10

def data_processing():

    with open(path+"data_batch_1", 'rb') as f1:
        dict1 = cPickle.load(f1)
    with open(path+"data_batch_2", 'rb') as f2:
        dict2 = cPickle.load(f2)
    with open(path+"data_batch_3", 'rb') as f3:
        dict3 = cPickle.load(f3)
    with open(path+"data_batch_4", 'rb') as f4:
        dict4 = cPickle.load(f4)
    with open(path+"data_batch_5", 'rb') as f5:
        dict5 = cPickle.load(f5)

    all_data=np.concatenate((dict1['data'],dict2['data'],dict3['data'],dict4['data'],dict5['data']),axis=0)
    all_label=np.concatenate((dict1['labels'],dict2['labels'],dict3['labels'],dict4['labels'],dict5['labels']),axis=0)
    all_dataset=zip(all_data,all_label)

    '''class method1'''
    '''CLASS=[]
    for i in xrange(10):
        CLASS.append(list())'''

    '''class method 2'''
    CLASS = [ []  for C in range(10)]

    '''Separate by class.'''
    for i in xrange(len(all_dataset)):
        #airplane
        if all_dataset[i][1] == 0:
            CLASS[0].append(all_dataset[i][0])
        # automobile
        if all_dataset[i][1] == 1:
            CLASS[1].append(all_dataset[i][0])
        # bird
        if all_dataset[i][1] == 2:
            CLASS[2].append(all_dataset[i][0])
        # cat
        if all_dataset[i][1] == 3:
            CLASS[3].append(all_dataset[i][0])
        # deer
        if all_dataset[i][1] == 4:
            CLASS[4].append(all_dataset[i][0])
        # dog
        if all_dataset[i][1] == 5:
            CLASS[5].append(all_dataset[i][0])
        # frog
        if all_dataset[i][1] == 6:
            CLASS[6].append(all_dataset[i][0])
        # horse
        if all_dataset[i][1] == 7:
            CLASS[7].append(all_dataset[i][0])
        # ship
        if all_dataset[i][1] == 8:
            CLASS[8].append(all_dataset[i][0])
        # truck
        elif all_dataset[i][1] == 9:
            CLASS[9].append(all_dataset[i][0])

    '''1. all data'''
    #return np.array(all_data).reshape(len(all_data),3,32,32)
    '''airplane, automobile , bird , cat , deer , dog , frog , horse , ship ,truck
       CLASS[0] , CLASS[1]   ...          CLASS[4]        ...                CLASS[9]'''

    '''2. class by class data'''
    return np.array(CLASS[7]).reshape(len(CLASS[7]),3,32,32) #horse

if __name__ == "__main__":
    "Load the cifar10 data from inside"
    data=data_processing()
    data =data.transpose((0, 2, 3, 1))
    '''visualization'''
    fig ,  ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig.suptitle('CIFAR10')
    for j in xrange(row_size):
        for i in xrange(column_size):
            ax[j][i].set_axis_off()
            ax[j][i].imshow(data[i+j*column_size])
    plt.show()
else:
    print "Load the cifar10 data from the outside"
