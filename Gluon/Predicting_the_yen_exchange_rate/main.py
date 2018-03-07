import mxnet as mx
cell_type = "GRU"

if cell_type=="LSTM":
    import LSTM_Cell
    LSTM_Cell.exchange_rate_model(epoch=0, time_step=28, day=1, normalization_factor=100, save_period=500 , load_period=500 , learning_rate=0.0001, ctx=mx.gpu(0))
elif cell_type=="GRU":
    import GRU_Cell
    GRU_Cell.exchange_rate_model(epoch=0, time_step=28, day=1, normalization_factor=100, save_period=500 , load_period=500 , learning_rate=0.0001, ctx=mx.gpu(0))
else :
    print("please write the cell type exactly")
