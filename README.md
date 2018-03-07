
>## ***Mxnet?*** 
* ***Flexible and Efficient Library for Deep Learning***
* ***Symbolic programming or Imperative programming***
* ***Mixed programming available*** ***(`Symbolic + imperative`)***
 
<image src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png" width=800 height=200></image>
>## ***Introduction*** 
*   
    It is a tutorial that can be helpful to those `who are new to the MXNET Deep-Learning Framework`
>## ***Official Homepage Tutorial***
*
    The following LINK is a tutorial on the MXNET  official homepage
    * `Link` : [mxnet homapage tutorials](http://mxnet.incubator.apache.org/tutorials/index.html)
>## ***Let's begin with***
* Required library and very simple code
```python
import mxnet as mx
import numpy as np

out=mx.nd.ones((3,3),mx.gpu(0))
print(mx.asnumpy(out))
```
* The below code is the result of executing the above code
```
<NDArray 3x3 @gpu(0)>
[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]
```      
>## ***Topic 1 : Symbolic Programming***

* ### ***Neural Networks basic with <Symbol.API + Optimizer Class>***
    ```python
    The following code is the most basic mxnet 'symbolic programming' technique using only 'Symbol.API', 'Symbol.bind Function', and Optimizer classes. It is very flexible. If you only understand the code below, you can implement whatever you want.
    ```
    * [***K-means Algorithm Using Random Data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/k_means)
        ```python
        I implemented 'k-means algorithm' for speed comparison.

        1. 'kmeans.py' is implemented using partial symbol.API.(Why partial? Only the assignment part of k-means algorithm was implemented with symbol.API)

        Comparison between symbol and ndarray(see below 'NDArray.API' for more information)
        -> As the number of data increases, symbol is a little faster than ndarray.

        Comparison between cpu and gpu
        -> As the number of data increases, gpu-symbol is overwhelmingly faster than cpu-symbol.
        ```
    * [***Very flexible Fully Connected Neural Network with Symbol.API , Dictionary parameter : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Very%20flexible%20Fully%20Connected%20Neural%20Network1)

    * [***Very flexible Fully Connected Neural Network with Symbol.API , List parameter : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Very%20flexible%20Fully%20Connected%20Neural%20Network2)
    * [***Very flexible Autoencoder Neural Networks with Symbol.API , Dictionary parameter : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Very%20flexible%20Autoencoder%20Neural%20Networks)

    * [***Very flexible Convolutional Neural Networks with Symbol.API , Dictionary parameter : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Very%20flexible%20Convolutional%20Neural%20Networks)

    * [***Very flexible Recurrent Neural Networks with Symbol.API , Dictionary parameter : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Very%20flexible%20Recurrent%20Neural%20Networks)

        * [***Modified Recurrent Neural Networks Cell by JG***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/CustomizedRNNCELL)
            ```python
            Please refer to this code when you want to freely transform 'shape' or 'structure' of 'RNN Cell', 'LSTM Cell', 'GRU Cell'. For example, it can be used to implement a transformation structure such as 'sequence to sequence'.
            ```

    * [***Very flexible Fully Connected 'Sparse' Neural Network with Sparse Symbol.API : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Very%20flexible%20Fully%20Connected%20Sparse%20Neural%20Network)
        ```python
        Train flexible Fully Connected Neural Network with Sparse Symbols
        ```

* ### ***Neural Networks basic with <Symbol.API + Module.API>***

    ```python
    The following code is a high-level interface Using 'Symbol.API' and 'Module.API'. It is fairly easy and quick to implement a formalized neural network. However, if you are designing a flexible neural net, do not use it.
    ```

    * [***Fully Connected Neural Network with LogisticRegressionOutput : Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Neural%20Network%20with_LogisticRegressionOutput)

    * [***Fully Connected Neural Network with SoftmaxOutput : Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Neural%20Network%20with_softmax)
    
    * [***Fully Connected Neural Network with SoftmaxOutput*** *(flexible)* : ***Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Neural%20Network%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Convolutional Neural Networks with SoftmaxOutput : Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput)

    * [***Convolutional Neural Networks with SoftmaxOutput*** *(flexible)* : ***Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Recurrent Neural Networks with SoftmaxOutput : Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20with%20SoftmaxOutput)
    
    * [***Recurrent Neural Networks + LSTM with SoftmaxOutput : Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20LSTM%20with%20SoftmaxOutput)

    * [***Recurrent Neural Networks + LSTM with SoftmaxOutput*** *(flexible)* : ***Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20LSTM%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module)) 
    
    * [***Recurrent Neural Networks + GRU with SoftmaxOutput : Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20GRU%20with%20SoftmaxOutput)

    * [***Recurrent Neural Networks + GRU with SoftmaxOutput*** *(flexible)* : ***Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20GRU%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Autoencoder Neural Networks with logisticRegressionOutput : Using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Autoencoder%20Neural%20Networks%20with%20logisticRegressionOutput)

    * [***Autoencoder Neural Networks with logisticRegressionOutput*** *(flexible)* : ***Using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Autoencoder%20Neural%20Networks%20with%20logisticRegressionOutput(flexible%20to%20use%20the%20module))

    * [***Fully Connected 'Sparse' Neural Network with SoftmaxOutput*** *(flexible)* : ***Classifying MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Sparse%20Neural%20Network%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))
        ```python
        Train flexible Fully Connected Neural Network with Sparse Symbols
        ```

* ### ***Neural Networks with visualization***
    * [***mxnet with graphviz library***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/visualization/visualization)
        ```cmd
        <linux>
        pip install graphviz(in anaconda Command Prompt) 

        <window>
        1. download 'graphviz-2.38.msi' at 'http://www.graphviz.org/Download_windows.php'
        2. Install to 'C:\Program Files (x86)\'
        3. add 'C:\Program Files (x86)\Graphviz2.38\bin' to the environment variable PATH
        ```
        ```python
        '''<sample code>'''  
        import mxnet as mx  

        '''neural network'''
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        # first_hidden_layer
        affine1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=50)
        hidden1 = mx.sym.Activation(data=affine1, name='sigmoid1', act_type="sigmoid")

        # two_hidden_layer
        affine2 = mx.sym.FullyConnected(data=hidden1, name='fc2', num_hidden=50)
        hidden2 = mx.sym.Activation(data=affine2, name='sigmoid2', act_type="sigmoid")

        # output_layer
        output_affine = mx.sym.FullyConnected(data=hidden2, name='fc3', num_hidden=10)
        output=mx.sym.SoftmaxOutput(data=output_affine,label=label)

        # Create network graph
        graph=mx.viz.plot_network(symbol=output)
        graph.view() # Show graphs and save them in pdf format.
        ```
    * [***MXNET with Tensorboard Only Available On Linux - Currently Only Works In Python 2.7***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/visualization/tensorboard-linux)
        
        ```python
        pip install tensorboard   
        ```
        ```python
        '''Issue'''
        The '80's line of the tensorboard file in the path '/home/user/anaconda3/bin' should be modified as shown below.
        ```
        ```python
        <code>
        for mod in package_path:
            module_space = mod + '/tensorboard/tensorboard' + '.runfiles'
            if os.path.isdir(module_space):
                return module_space
        ```
        * If you want to see the results immediately,`write the following script in the terminal window` where the event file exists.
        
            * `tensorboard --logdir=tensorboard --logdir=./ --port=6006`

* ### ***Neural Networks Applications***

    * [***Predicting lotto numbers in regression analysis***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/Predicting%20lotto%20numbers%20in%20regression%20analysis%20using%20mxnet)

    * [***Generative Adversarial Networks with fullyConnected Neural Network : Using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/Generative%20Adversarial%20Network%20with%20FullyConnected%20Neural%20Network)

    * [***Deep Convolution Generative Adversarial Network : Using ImageNet , CIFAR10 , MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/Deep%20Convolution%20Generative%20Adversarial%20Network)
        ```cmd
        <Code execution example>  
        python main.py --state --epoch 100 --noise_size 100 --batch_size 200 --save_period 100 --dataset CIFAR10 --load_weights 100
        ```
    * [***Neural Style with Symbol.API***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/NeuralStyle)
        ```cmd
        to configure a network of flexible structure, use 'bind function' of Symbol.API and 'optimizer class'. - See the code for more information!!!
        ```
    * [***Dynamic Routing Between Capsules : Capsule Network only using Symbol.API***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/CapsuleNet_Symbol)

     * [***Dynamic Routing Between Capsules : Capsule Network using Module.API***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/CapsuleNet_Module)

    * [***Variational Autoencoder : Using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/)
    
* ### ***How to Create New Operators for Symbolic? - Advanced***
    * #### [References Page : It is insufficient.](https://mxnet.incubator.apache.org/how_to/new_op.html) 
        
        * [***Customized LogisticRegressionOutput Layer***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/advanced/Fully%20Connected%20Neural%20Network%20with%20Custom%20LogisticRegressionOutput)

        * [***Customized SoftmaxOutput Layer***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/advanced/Fully%20Connected%20Neural%20Network%20with%20Custom%20SoftmaxOutput)

        * [***Customized SoftmaxOutput + Customized Activation Layer***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/advanced/Fully%20Connected%20Neural%20Network%20with%20Custom%20Activation)


>## ***Topic 2 : Imperative Programming***

* ### ***Neural Networks With <NDArray.API + Autograd Package + Optimizer Class>***

    ```python
    The NDArray API is different from mxnet-symbolic coding.
    It is imperactive coding and focuses on NDArray of mxnet.
    ```

    * #### The following LINK is a tutorial on the [gluon page](http://gluon.mxnet.io/index.html)  official homepage

        * [***K-means Algorithm Using Random Data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/k_means)
            ```python
            I implemented 'k-means algorithm' in two ways for speed comparison.

            1. 'kmeans_numpy.py' is implemented using mxnet-ndarray
            2. 'kmeans.py' is implemented using numpy

            Comparison between ndarray and numpy
            -> As the number of data increases, ndarray is overwhelmingly faster than numpy.

            Using ndarray only
            Comparison between cpu and gpu
            -> As the number of data increases, gpu-ndarray is overwhelmingly faster than cpu-ndarray.
            ```

        * [***Multiclass logistic regression : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Multiclass_logistic_regression_with_NDArray)

        * [***Fully Neural Network : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Fully_Neural_Network_with_NDArray)

        * [***Convolution Neural Network : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_with_NDArray)

        * [***Convolution Neural Network With Batch Normalization(First Method) : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_BN1_with_NDArray)

        * [***Convolution Neural Network With Batch Normalization(second Method) : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_BN2_with_NDArray)

        * [***Convolution Neural Network With Builtin Batch Normalization : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_Using_builtin_BN_with_NDArray)

        * [***Autoencoder Neural Networks : Using MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Autoencoder_Neural_Network_with_NDArray)

        * [***Convolution Autoencoder Neural Networks : Using MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Autoencoder_Neural_Network_with_NDArray)

        * [***Recurrent Neural Network(RNN, LSTM, GRU) : Classifying Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Recurrent_Neural_Network_with_NDArray)

* ### ***Neural Networks With <Gluon Package + NDArray.API + Autograd Package>***

    ```python
    The Gluon package is different from mxnet-symbolic coding.
    It is imperactive coding and focuses on mxnet with NDArray and Gluon.

    - You can think of it as a deep learning framework that wraps NDARRAY.API easily for users to use.
    ```

    * #### The following LINK is a tutorial on the [gluon page](http://gluon.mxnet.io/index.html)  official homepage

        * [***K-means Algorithm Using Random Data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/k_means)
            ```python
            I implemented 'k-means algorithm' for speed comparison.

            1. 'kmeans.py' is implemented using Gluon package.

            Comparison between gluon and ndarray(see above 'NDArray.API' for more information)
            -> As the number of data increases, Gluon is a little bit faster than ndarray.

            Comparison between cpu and gpu
            -> As the number of data increases, gpu-gluon is overwhelmingly faster than cpu-gluon.
            ```

        * ***Using nn.Sequential when writing your `High-Level Code`***

            * [***Multiclass logistic regression : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Multiclass_logistic_regression_with_Gluon)

            * [***Fully Neural Network : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Fully_Neural_Network_with_Gluon)

            * [***Convolution Neural Network : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Neural_Network_with_Gluon)

            * [***Convolution Neural Network with Batch Normalization : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Neural_Network_BN_with_Gluon)

            * [***Autoencoder Neural Networks : Using MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Autoencoder_Neural_Network_with_Gluon)

            * [***Convolution Autoencoder Neural Networks : Using MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Autoencoder_Neural_Network_with_Gluon)

            * [***Recurrent Neural Network(RNN, LSTM, GRU) : Classifying Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Recurrent_Neural_Network_with_Gluon)
                ```python
                'Recurrent Layers' in 'gluon' are the same as FusedRNNCell in 'symbol.API'. 
                -> Because it is not flexible, Recurrent Layers is not used here.
                ```

        * ***Using Block when Designing a `Custom Layer` - Flexible use of Gluon***

            * [***Convolution Neural Network with Block or HybridBlock : Classifying MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Neural_Network_with_Block)
                ```python
                It is less flexible than Block in network configuration.
                ```
                * [More information on `Block` and `HybridBlock`](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html) 
        
* ### ***Neural Networks Applications with <NDArray.API + Gluon Package + Autograd Package>***

    * [***Predicting Lotto Numbers in Regression Analysis***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Predicting_lotto_numbers_in_regression_analysis_using_Gluon)

    * [***Neural Style with NDArray.API***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/NeuralStyle)
        ```cmd
        To configure a network of flexible structure, You should be able to set the object you want to differentiate. - See the code for more information!!!
        ```

    * [***Neural Style with Gluon Package***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/NeuralStyle)
        ```cmd
        To configure a network of flexible structure, You should be able to set the object you want to differentiate. - See the code for more information!!!
        ```

    * [***Deep Convolution Generative Adversarial Networks Using CIFAR10 , FashionMNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/DCGAN_with_Gluon)

    * [***Deep Convolution Generative Adversarial Networks Targeting Using CIFAR10 , MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/DCGAN_target_with_Gluon)

    * [***Predicting the yen exchange rate with LSTM or GRU***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Predicting_the_yen_exchange_rate)

        ```python
        Finding 'xxxx' - JPY '100' is 'xxxx' KRW 
        I used data from '2010.01.04' ~ '2017.11.25'
        ```
    * [***Dynamic Routing Between Capsules : Capsule Network***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/CapsuleNet)

    * [***Variational Autoencoder : Using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Variational%20Autoencoder)

>## ***Development environment***
* os : ```window 10.1 64bit``` and ```Ubuntu linux 16.04.2 LTS only for tensorboard``` 
* python version(`3.6.1`) : `anaconda3 4.4.0` 
* IDE : `pycharm Community Edition 2017.2.2 or visual studio code`
    
>## ***Dependencies*** 
* mxnet-1.0.0
* numpy-1.12.1, matplotlib-2.0.2 , tensorboard-1.0.0a7(linux) , graphviz -> (`Visualization`)
* tqdm -> (`Extensible Progress Meter`)
* opencv-3.3.0.10 , struct , gzip , os , glob , threading -> (`Data preprocessing`)
* Pickle -> (`Data save and restore`)
* logging -> (`Observation during learning`)
* argparse -> (`Command line input from user`)
* urllib , requests -> (`Web crawling`) 

>## ***Author*** 
[JONGGON KIM GitHub](https://github.com/JONGGON)
 
rlawhdrhs27@gmail.com