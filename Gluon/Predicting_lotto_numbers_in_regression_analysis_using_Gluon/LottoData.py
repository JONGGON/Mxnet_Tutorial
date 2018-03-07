# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import mxnet as mx

class LOTTO(mx.gluon.data.Dataset):

    def __init__(self,train=True):

        self.train=train
        self._data_preprocessing()

    def __repr__(self):
        return "'Lotto Dataset'"

    def __getitem__(self, idx):

        return self._data[idx], self._label[idx]

        '''if self.train==True:
            return self._data[idx], self._label[idx]
        else:
            return self._data[idx]'''

    def __len__(self):
        return len(self._data)

    def _data_preprocessing(self):


        if self.train==True:
            # lotto data
            data = pd.read_excel("lotto.xls")
            # change the type of data to numpy
            data = np.asarray(data)
            input = data[1:, 1:-1]
            output = data[0:np.shape(data)[0] - 1, 1:-1]
            self._data = np.flipud(input).astype(np.float32)
            self._label = np.flip(output, axis=0).astype(np.float32)

        else :
            self._data = np.array([[6 , 10 , 17 , 18 , 21 , 29], [ 5 , 6 , 11 , 14 , 21 , 41]],dtype=np.float32)
            self._label = np.zeros(np.shape(self._data)).astype(np.float32) # __getitem__' is required for proper operation. (It should be the same size as 'self._data' above.)

if __name__=="__main__":
    print(LOTTO())
else:
    print("Lotto Dataset imported")
