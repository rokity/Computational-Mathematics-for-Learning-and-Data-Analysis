# Contributions Document

## Reference to change project

- Batch and Epochs management ---> neural_network.py (fit function/method)
- Plot Images ---> neural_network.py
- Print Loss Accuracy ---> neural_network.py
- Kernel Initalization ---> kernel_initialization.py
- Feedforword formula ---> layer.py (feedforward function/method)
- Activation function ---> 

## File Description

### kernel_initialization.py
Algorithms that initiliaze weight matrix of neural network ,before to start to train the model.
If you want to change initialaziont weight matrix neural network you must to watch here.


### neural_network.py
This class is used to create a neural network. Train the model with training ,test and validation dataset. In this class you can find all the Management of Batch and Epochs so if you need is here. No feedforward, No backpropagation, No formulas.
If you need to alterate Plot Function, save Plot image, Print Verbose of training model you can find all the functions here.


### layer.py
Initialize the layer, create weight matrix and w0 matrix (self.b). Implement Feedforward and call Activation Function.
Print informationa about the layer. Save cache feedforward.


