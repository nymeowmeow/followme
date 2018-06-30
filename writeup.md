# RoboND - Follow me Project #

In this project, we will be building a semantic segmentation network using fully convolutional network to allow a drone to track and follow a single hero target.

[image_0]: ./images/network.JPG
[image_1]: ./images/epoch10.JPG
[image_2]: ./images/epoch50.JPG
[image_3]: ./images/model.JPG

## Architecture ##
![architecture][image_0] 

The network architecure used in this projected is as shown above. It is a typically fully convolutional network composed of encoder section followed by 1x1 convolution layer and finally a decoder section.

 the model is as follows
 
![model][image_3]

### encoder ###
The encoder section is a convolution network that reduces to a deeper 1x1 convolution layer, in contrast to a flat fully connected layer that would be used for basic classification of images.

The encoding section is using separable convolutions. Separable convolution also known as depthwise convolution comprise of convolution performed over each channel of an input layer and followed by 1x1 convolution that takes the output channel from the previous sep and then combined them into an output layer.

### 1x1 convolution layer ###
The 1x1 convolution layer is used to preserve spacial information from the image.

### decoder ###
The decoder section is used to upscale the output of the encoder, such that the resulting output is the same size as the original image. 

To upsample, we are using bilinear upsampling, which is a technique that utilizes the weighted average of neighboring points to extrapolate the value.

### skip connection ###
skip connection are used in the above network configuration, that allows subsequent layers to re-use output from early layer, which maintain more information which can lead to better performance.  

### output layer ###
The output layer is just a softmax layer, to predict the probability for each class.

## hyperparameters ##

The hyperparameter used is as follows

parameter | value
--------- | -------
learning rate | 0.0007
batch size | 64
num epochs | 50
steps per epoch | 72
validation steps | 50
workers | 2

It seems the learning rate needs to set to a small value. Initially i set the learning rate to be 0.01 but the performance is unstable and varies a lot. setting it to lower value improves the performance, eventually i settle on using learning rate of 0.0007.

initially i have the batch size to be around 100 and steps per epoch of 250, but it is taking a long time, so i trim it down to use batch size of 64 and steps per epoch around 72, so that batch size * steps per epoch is about the size of the training data used.

Also as i tried to increase the num of epochs from 10 to higher value, it run out of memory when i have batch size of 100 and steps per epoch 200, so i have to adjust the batch size and steps per epoch.

## Training ##
train/val loss after 10 

![epoch 10][image_1] 

train/val loss after 50 epochs

![epoch 50][image_2] 

## Result ##

The final IOU is 0.56, and final score is 0.415

### Model extension ###
This model should be able to use to follow other types of objects, like dog, cat or car etc. As long as we provide relevant training set, to train the system to recognise and detect that object class.

The model should not be limited to follow human, the model is based on semantic segmentation which works on pixel level, as long as the model is trained with proper training set, it should be able to detect and classify proper objects.

## Improvements
* adding more data should give better result. I tried to add some extra images to the dataset, but the result doesn't seem to have any significant effects. Probably what i have collected is too small a dataset, it needs more data before it show significant improvement.
* will be interesting to train the network to follow a car or other objects.



