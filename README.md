# LB-CNN
LB-CNN: Convolutional Neural Network with Latent Binarization for Large Scale Multi-class Classification

![alt text](https://github.com/treese41528/LB-CNN/blob/main/Images/LB_CNN.PNG)

# Snapshot of learned latent structure

![alt text](https://github.com/treese41528/LB-CNN/blob/main/Images/WorkingDogs.png)


# Model Files
The LB-CNN model files are stored under LB_CNN_Models. These files can be used as directly with input features for simple data or used with in combination with any neural network architecture. For use with a neural network instantiate the model with the required parameters and feed the output prior to the classification layer of a neural network to the models forward function, the return value is the log-likelihood of the model.



# Directions

To run the experiments install the contents of this repository onto your google drive under a folder called ColabNotebooks/LB-CNN.

Store the corresponding datasets under the directories ColabNotebooks/Data/MNIST, ColabNotebooks/Data/FashionMNIST, ColabNotebooks/Data/Cifar-10, ColabNotebooks/Data/Cifar-100.
Download Locations:
MNIST: http://yann.lecun.com/exdb/mnist/
FashionMNIST: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
Cifar10: https://www.cs.toronto.edu/~kriz/cifar.html
Cifar100: https://www.cs.toronto.edu/~kriz/cifar.html

Running Google Colabatory Notebooks:
You need to mount the google drive which requires giving permission as in the example https://colab.research.google.com/notebooks/io.ipynb.
The provided google collabatory notebooks already have the code for doing this provided you used the correct file structure provided above.
