#algorithm2 - A version that does not use 'for statement' to get 'style loss'.
import mxnet as mx
import urllib
import os


def VGG19(image):

    #vgg19 - convolution part
    data = mx.sym.Variable(image)
    conv1_1 = mx.symbol.Convolution(name=image+'conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu1_1 = mx.symbol.Activation(name=image+'relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name=image+'conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu1_2 = mx.symbol.Activation(name=image+'relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name=image+'pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')

    conv2_1 = mx.symbol.Convolution(name=image+'conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_1 = mx.symbol.Activation(name=image+'relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name=image+'conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_2 = mx.symbol.Activation(name=image+'relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name=image+'pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')

    conv3_1 = mx.symbol.Convolution(name=image+'conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_1 = mx.symbol.Activation(name=image+'relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name=image+'conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_2 = mx.symbol.Activation(name=image+'relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name=image+'conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_3 = mx.symbol.Activation(name=image+'relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name=image+'conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_4 = mx.symbol.Activation(name=image+'relu3_4', data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(name=image+'pool3', data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv4_1 = mx.symbol.Convolution(name=image+'conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_1 = mx.symbol.Activation(name=image+'relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name=image+'conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name=image+'relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name=image+'conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_3 = mx.symbol.Activation(name=image+'relu4_3', data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name=image+'conv4_4', data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_4 = mx.symbol.Activation(name=image+'relu4_4', data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(name=image+'pool4', data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')

    conv5_1 = mx.symbol.Convolution(name=image+'conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu5_1 = mx.symbol.Activation(name=image+'relu5_1', data=conv5_1 , act_type='relu')

    # style and content layers selection
    style = mx.sym.Group([relu1_1, relu2_1, relu3_1, relu4_1, relu5_1])
    content = mx.sym.Group([relu4_2])
    return style, content

def algorithm(content_a=1, style_b=1, content_image=None, style_image=None, noise_image=None ,image_size=(256,512), ctx=mx.gpu(0)):

    _, content = VGG19("content_")
    style , _  = VGG19("style_")
    noise_style , noise_content = VGG19("noise_")

    #The group order is important for the code below.
    group = mx.sym.Group([content , style , noise_content, noise_style])
    #print(group)

    #(1) Get the name of the 'argument'
    #content_ , style_ , noise_content , noise_style
    arg_names = group.list_arguments()
    arg_shapes, output_shapes, aux_shapes = group.infer_shape(content_=(1, 3, image_size[0], image_size[1]),style_=(1, 3, image_size[0], image_size[1]),noise_=(1, 3, image_size[0], image_size[1]))

    #(2) Make space for 'argument'
    arg_dict= dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    arg_dict["content_"]=content_image
    arg_dict["style_"]=style_image
    '''
    args and args_grad must have their own spaces.

    args : list of NDArray or dict of str to NDArray
        Input arguments to the symbol.

        - If the input type is a list of `NDArray`, the order should be same as the order
          of `list_arguments()`.
        - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
          to the corresponding `NDArray`.
        - In either case, all the arguments must be provided.

    args_grad : list of NDArray or dict of str to `NDArray`, optional
        When specified, `args_grad` provides NDArrays to hold
        the result of gradient value in backward. 

        - If the input type is a list of `NDArray`, the order should be same as the order
          of `list_arguments()`.
        - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
          to the corresponding NDArray.
        - When the type is a dict of str to `NDArray`, one only need to provide the dict
          for required argument gradient.
          Only the specified argument gradient will be calculated.
    '''
    arg_dict["noise_"]=noise_image
    grad_dict=dict(noise_=mx.nd.zeros(shape=arg_dict["noise_"].shape, ctx=ctx))

    #(4) compute content loss using  cov4_2
    batch_size, filter, height, width = output_shapes[0] # or output_shapes[6]
    c = group[0].reshape(shape=(-1,height*width))
    n = group[6].reshape(shape=(-1,height*width))

    #content loss
    content_loss=0.5*mx.sym.square(n-c)
    content_loss=mx.sym.mean(content_loss)

    #(5) compute style loss using cov1_1 ,cov2_1 ,cov3_1 ,cov4_1 ,cov5_1

    batch_size, filter, height, width = output_shapes[7] # output_shapes[1]

    # reshape
    n = group[7].reshape((-1, height * width))
    s = group[1].reshape((-1, height * width))
    N = filter
    M = height * width
    # gram_matrix
    n = mx.sym.dot(n, n, transpose_a=False, transpose_b=True)  # (filter, filter)
    s = mx.sym.dot(s, s, transpose_a=False, transpose_b=True)  # (filter, filter)'''

    style_loss1 = mx.sym.mean((mx.sym.square(n - s) / (4 * N * M)) * 0.2)
    #style_loss1 = mx.sym.mean((mx.sym.square(n-s)/(4 * (N*N) * (M*M)))*0.2) # nd.mean((filter,))

    batch_size, filter, height, width = output_shapes[8]# output_shapes[2]
    # reshape
    n = group[8].reshape((-1, height * width))
    s = group[2].reshape((-1, height * width))
    N = filter
    M = height * width

    # gram_matrix
    n = mx.sym.dot(n, n, transpose_a=False, transpose_b=True)  # (filter, filter)
    s = mx.sym.dot(s, s, transpose_a=False, transpose_b=True)  # (filter, filter)'''

    style_loss2 = mx.sym.mean((mx.sym.square(n - s) / (4 * N * M)) * 0.2)
    #style_loss2 = mx.sym.mean((mx.sym.square(n-s)/(4 * (N*N) * (M*M)))*0.2) # nd.mean((filter,))

    batch_size, filter, height, width = output_shapes[9]# output_shapes[3]
    # reshape
    n = group[9].reshape((-1, height * width))
    s = group[3].reshape((-1, height * width))
    N = filter
    M = height * width
    # gram_matrix
    n = mx.sym.dot(n, n, transpose_a=False, transpose_b=True)  # (filter, filter)
    s = mx.sym.dot(s, s, transpose_a=False, transpose_b=True)  # (filter, filter)'''

    style_loss3 = mx.sym.mean((mx.sym.square(n - s) / (4 * N * M)) * 0.2)
    #style_loss3 = mx.sym.mean((mx.sym.square(n-s)/(4 * (N*N) * (M*M)))*0.2) # nd.mean((filter,))

    batch_size, filter, height, width = output_shapes[10]# output_shapes[4]
    # reshape
    n = group[10].reshape((-1, height * width))
    s = group[4].reshape((-1, height * width))
    N = filter
    M = height * width
    # gram_matrix
    n = mx.sym.dot(n, n, transpose_a=False, transpose_b=True)  # (filter, filter)
    s = mx.sym.dot(s, s, transpose_a=False, transpose_b=True)  # (filter, filter)'''

    style_loss4 = mx.sym.mean((mx.sym.square(n - s) / (4 * N * M)) * 0.2)
    #style_loss4 = mx.sym.mean((mx.sym.square(n-s)/(4 * (N*N) * (M*M)))*0.2) # nd.mean((filter,))

    batch_size, filter, height, width = output_shapes[10]# output_shapes[5]
    # reshape
    n = group[11].reshape((-1, height * width))
    s = group[5].reshape((-1, height * width))
    N = filter
    M = height * width
    # gram_matrix
    n = mx.sym.dot(n, n, transpose_a=False, transpose_b=True)  # (filter, filter)
    s = mx.sym.dot(s, s, transpose_a=False, transpose_b=True)  # (filter, filter)'''

    style_loss5 = mx.sym.mean((mx.sym.square(n-s)/(4 * N * M))*0.2) # nd.mean((filter,)) # nd.mean((filter,))
    #style_loss5 = mx.sym.mean((mx.sym.square(n-s)/(4 * (N*N) * (M*M)))*0.2) # nd.mean((filter,))

    style_loss=style_loss1+style_loss2+style_loss3+style_loss4+style_loss5
    total_loss=mx.sym.MakeLoss(data=(content_a*content_loss+style_b*style_loss),grad_scale=1)

    # We visualize the network structure with output size (the batch_size is ignored.)
    graph = mx.viz.plot_network(symbol=total_loss)  # The diagram can be found on the Jupiter notebook.
    graph.view()

    #(5) How to get pretrained model from mxnet 'symbol' - VGG19
    if os.path.exists("vgg19.params"):
        print("vgg19.params exists")
        pretrained = mx.nd.load("vgg19.params")
    else:
        print("vgg19.params downloading")
        url="http://data.dmlc.ml/models/imagenet/vgg/vgg19-0000.params"
        urllib.request.urlretrieve(url,"vgg19.params")
        print("vgg19.params downloading completed")
        pretrained = mx.nd.load("vgg19.params")

    for name in arg_names:
        if name == "content_" or name == "style_" or name=="noise_":
             continue
        rename=name.split("_")
        key = "arg:" + rename[1]+"_"+rename[2]+"_"+rename[3]
        #print(key)
        if key in pretrained:
            arg_dict[name]=pretrained[key].as_in_context(ctx)
        else:
          print("Skip argument {}".format(name))

    return total_loss.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")

#If you are wondering about bind, read the explanation below.
"""Binds the current symbol to an executor and returns it.

We first declare the computation and then bind to the data to run.
This function returns an executor which provides method `forward()` method for evaluation
and a `outputs()` method to get all the results.

Example
-------
>>> a = mx.sym.Variable('a')
>>> b = mx.sym.Variable('b')
>>> c = a + b
<Symbol _plus1>
>>> ex = c.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3]), 'b' : mx.nd.ones([2,3])})
>>> ex.forward()
[<NDArray 2x3 @cpu(0)>]
>>> ex.outputs[0].asnumpy()
[[ 2.  2.  2.]
[ 2.  2.  2.]]

Parameters
----------
ctx : Context
    The device context the generated executor to run on.

args : list of NDArray or dict of str to NDArray
    Input arguments to the symbol.

    - If the input type is a list of `NDArray`, the order should be same as the order
      of `list_arguments()`.
    - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
      to the corresponding `NDArray`.
    - In either case, all the arguments must be provided.

args_grad : list of NDArray or dict of str to `NDArray`, optional
    When specified, `args_grad` provides NDArrays to hold
    the result of gradient value in backward. 

    - If the input type is a list of `NDArray`, the order should be same as the order
      of `list_arguments()`.
    - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
      to the corresponding NDArray.
    - When the type is a dict of str to `NDArray`, one only need to provide the dict
      for required argument gradient.
      Only the specified argument gradient will be calculated.

grad_req : {'write', 'add', 'null'}, or list of str or dict of str to str, optional
    To specify how we should update the gradient to the `args_grad`.

    - 'write' means everytime gradient is write to specified `args_grad` `NDArray`.
    - 'add' means everytime gradient is add to the specified NDArray.
    - 'null' means no action is taken, the gradient may not be calculated.

aux_states : list of `NDArray`, or dict of str to `NDArray`, optional
    Input auxiliary states to the symbol, only needed when the output of
    `list_auxiliary_states()` is not empty.

    - If the input type is a list of `NDArray`, the order should be same as the order
      of `list_auxiliary_states()`.
    - If the input type is a dict of str to `NDArray`, then it maps the name of
      `auxiliary_states` to the corresponding `NDArray`,
    - In either case, all the auxiliary states need to be provided.

group2ctx : Dict of string to mx.Context
    The dict mapping the `ctx_group` attribute to the context assignment.

shared_exec : mx.executor.Executor
    Executor to share memory with. This is intended for runtime reshaping, variable length
    sequences, etc. The returned executor shares state with `shared_exec`, and should not be
    used in parallel with it.

Returns
-------
executor : Executor
    The generated executor

Notes
-----
Auxiliary states are the special states of symbols that do not correspond
to an argument, and do not have gradient but are still useful
for the specific operations. Common examples of auxiliary states include
the `moving_mean` and `moving_variance` states in `BatchNorm`.
Most operators do not have auxiliary states and in those cases,
this parameter can be safely ignored.

One can give up gradient by using a dict in `args_grad` and only specify
gradient they interested in.
"""