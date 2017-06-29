#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Nvidia_model.png "Nvidia model"
[image2]: ./images/placeholder.png "Grayscaling"
[image3]: ./images/placeholder_small.png "Recovery Image"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 168-199)

The model includes ELU layers to introduce nonlinearity (code line 173, 175, 177, 179, 181), and the data is normalized in the model using a Keras lambda layer (code line 172).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 184,186,188).

The model was trained `generator()` (code line 51-111) and validated `generator_val()` on different data sets to ensure that the model was not overfitting (code line 113-145) . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in the opposite direction to balance the number of left and right turns.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build the model incrementally.

My first step was to use a convolution neural network model similar to the NVIDIA end-to-end model, I thought this model might be appropriate because it shows great results on driving on actual car, also I found it simple to understand.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or not driving right, to improve the driving behavior in these cases, I retrained the model on recovery data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]
1. 5 convolutional2d layers increase the feature depth (see image for hyperparameters).
1. Relu activations find non-linear relationships between layers.
1. Three fully connected layers added at end, eventually outputing a steering angle.
1. Using adam optimizer, mean squared error minimized (distance from predicted steering angle and actual).
1. 48k images (after processing) trained for 15 epochs with batch size 128 at .001, then for a couple more at .0001

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on track by itself. These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help the model generalize better. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


Before collection I had 6174 data point, after the collection process, I had 18522 number of data points. I then preprocessed this data by cropping the images to 64x64 since it doesn't reduce the model accuracy but speed up the training.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 maybe less as evidenced the validation accuracy stopped improving after 10 epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
