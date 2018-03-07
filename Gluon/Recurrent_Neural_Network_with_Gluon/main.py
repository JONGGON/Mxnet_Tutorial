import mxnet as mx
cell_type = "RNN"

#dataset = FashionMNIST
if cell_type=="RNN":
    import RNN
    RNN.RNN(epoch=30, batch_size=256 , save_period=30 , load_period=30 , learning_rate=0.001, ctx=mx.gpu(0))
elif cell_type=="LSTM":
    import LSTM
    LSTM.LSTM(epoch=30, batch_size=256 , save_period=30 , load_period=30 , learning_rate=0.001, ctx=mx.gpu(0))
elif cell_type=="GRU":
    import GRU
    GRU.GRU(epoch=30, batch_size=256 , save_period=30 , load_period=30 , learning_rate=0.001, ctx=mx.gpu(0))
else :
    print("please write the cell type exactly")
