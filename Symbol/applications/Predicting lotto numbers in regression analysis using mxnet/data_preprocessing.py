# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#데이터 읽어오기
def data_preprocessing():

    #data normalization
    normalization_factor=1.0
    Lotto_number=6

    #show all data
    np.set_printoptions(threshold=1000) #threshold 총갯수

    #lotto data
    data = pd.read_excel("lotto.xls")

    #change the type of data to numpy
    data=np.asarray(data)

    #data normalization
    generator=data[:,1:-1]

    input =  data[1: , 1:-1]/normalization_factor
    output = data[0:np.shape(data)[0]-1,1:-1]/normalization_factor

    #flip the lotto data - two method
    generator=np.flipud(generator)
    input=np.flipud(input)
    output = np.flip(output, axis=0)
    #version1
    #training_data=zip(generator,generator)

    #version2
    training_data=zip(input,output)

    #test_data
    test_data=np.array([[5,15,20,31,34,42],[7,27,29,30,38,44]]).reshape(-1,Lotto_number)/normalization_factor
    return training_data,test_data,normalization_factor

if __name__=="__main__":
    print("data_preprocessing_starting in main")
    data_preprocessing()
else:
    print("data_preprocessing_imported")

