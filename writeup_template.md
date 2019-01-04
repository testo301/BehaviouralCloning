# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/lossplot.jpg "Train/Validation loss versus number of epochs"

[image2]: ./images/model_architecture1.JPG "Model Architecture"
[image3]: ./images/model_architecture2.JPG "Model Architecture"

[image4]: ./images/cropped.jpg "Cropped Image"
[image5]: ./images/flipped.jpg "Flipped Image"
[image6]: ./images/leftcenterright.jpg "Left / Right / Center Image"
[image7]: ./images/steeringhistogram.jpg "Histogram of the steering angles"





## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_template.md and writeup_template.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 

The code contains three sections:
1. Definition of the generator function generator() taking as an argument the sample, along with the size of the batch
2. Importing samples from the .csv files
3. Defining architecture of the network with Keras
4. Compiling the model, definig checkpoints and executing the training


### Model Architecture and Training Strategy

#### 0. Approach strategy

Given large complexity of the problem in terms of number of parameters, it was impossible to isolate the factors and ensure proper experimentation is done to optimize all aspects of the project, that is:
- input image choice (key turns only, all track, multiple takes of the track, backwards track, recoveries, etc.)
- vehicle speed versus steering angle effects
- input image size and cropping area
- color palette transformation
- random shadowing
- image flipping
- correction of the steering angle for the left/right images
- smoothing of the steering angles
- addressing assymetries of the steering angle distribution
- batch size
- number of epochs
- network depths
- network layer structure
- size of the convolution filters
- stack of the filters
- dropout layers
- pooling layers
- activation functions
- size of the fully connected layers
- method of convergence
- stopping criteria
- optimizer

Given limited timeframe, quick experiments were performed to choose:
- data collection - the simulator was downloaded locally, given bad performance of the simulator in the workspace, and the whole data collection exercise was performed locally for:
    - small recordings of only the key turns
    - full forward pass of the track
    - full backward pass of the track
    - additional pass and recovery recording of the challenging turn (where the simulator had tendency to approach the right lane)
- vehicle speed versus steering angles - I assumed vehicle drives full throttle, I didn't differentiate recordings in terms of vehicle speed and the impact of the steering angle for simplicity
- smoothing of the steering angle - I did not apply any smoothing to the steering angles. However I observed a direct impact of me driving more smoothly around easier curves and jerky movements around tighter turns on the behaviour of the network. The network inherited smoother way of driving from my forward pass, along with the jerky corrections in the backward pass. All forward/backward/curves training data available on demand since learning was performed locally on my machine.
- batch size - 32 proved to be a good choice, larger values increased severalfold the per epoch training time
- number of epochs - loss value was observed during the training and the full models were saved at the end of each epochs
- data augmentation - flipping of only the center image was applied. Expert correction of +/-0.25 was added as a scalar to the steering angles for the left/right images. 0.30 works fine as well. It can be derived. No other augmentation was applied given long training times for data.
- addressing assymetries of the steering angle distribution - image flipping helped to address assymetries. However it didn't focus on smoothing the distribution.
- no image distortion correction was applied.
- network architecture testing proved to be most challenging, minimalistic network didn't work on small data samples, therefore a proven networks were chosen NVIDIA's and Comma.AI's.
- optimization - Adam was chosen, with the MSE criteria. The loss value was observed during the pass over the respective epochs.

#### 1. An appropriate model architecture has been employed

Several model architectures were tested on a small data sample covering only key turns of the track.

1. Minimalistic model with Input -> 5x5 convolutional layer -> RELU activation -> Flattening -> 100 Dense Layer -> 1 Output 
2. NVIDIA model, as described in the well known paper
[End to End DL](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

3. Architecture proposed by https://comma.ai/ company, building on similar concept to the NVIDIA model. 
[Self Steering GH](https://github.com/commaai/research/blob/master/SelfSteering.md) related to the paper arxiv.org/abs/1608.01230


After checking the articles recommended in the submission of the traffic light project, ELU activation proves to be a better choice than leaky RELU.

1. The first minimalistic approach didn't tackle more complex curves appropriately.
2. The second choice took a long time to train the first epoch on the small data sample.
3. The third choice provided a model that trained through the first epoch in a reasonable time and readily provided a model that was able to drive itself (except stepping on a right lane line and then recovering on one of the more complex turns). 

Given tangible results after the first quick training round, the third Comma.AI based model was chosen.

The model consists of the convolutional layers with varying filter sizes, ELU activation layers, fully connected layers and the dropout layers with varying probability levels. The data is normalized in the model using a Keras lambda layer

Model architecture illustrated in the flow chart:

![alt text][image3]

The model architecture with dimensions as the extract from model.summary() is provided below:

![alt text][image2]

The architecture assumes 4, 2 and 2 strides in the convolutional layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The dropout layers 

The model was trained on three independently collected datasets:
- full forward pass through the track
- full backward pass through the track
- recovery of the trickier turn

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model is optimized with Adam optimizer, therefore no manual intervention was required.

#### 4. Appropriate training data

The model was trained on three independently collected datasets (collected on the local machine due to the performance of Workspace simulator):
- full forward pass through the track
- full backward pass through the track
- recovery of the trickier turn

All three camera views were used with the expertly corrected steering angle for the left and right camera.

The initial forward run through the track proved to have largely biased steering angles to one side which would prevent the model for successfully generalizing. 

My driving in the backward run was very nervous which is reflected by not-very-progressive steering. This nervousness was then passed on to the automatic driving in the final model. But it's interesting how it was inherited, so I left it there.

The following histogram illustrates the problem for the forward run (blue) and backward run (red) through the track.

![alt text][image7]


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first approach involved testing the minimalistic network of a single convolutional layer and a single fully connected layer on the smallest data sample possible (covering only key turns). I wanted to perform data collection + transformation + learning + simulation within 1 hour. Unfortunately the car was not able to properly clear the turns at the bifurcation of the offroad part.

Therefore I searched for a proven model, the following being the candidates that are relatively up-to-date (year 2016).
-NVIDIA model, as described in the well known paper
[End to End DL](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

- Architecture proposed by https://comma.ai/ company, building on similar concept to the NVIDIA model. 
[Self Steering GH](https://github.com/commaai/research/blob/master/SelfSteering.md) related to the paper arxiv.org/abs/1608.01230

The first model was resource hungry in terms of training, therefore I have selected the second one, given reasonable per epoch training time.

After each epoch, the model was saved and I was able to test its performance in the Workspace (unfortunately I didn't manage to run Autonomous model on the offline local simulator).

Model after the first epoch on the forward training data only was reasonably good. Except one curve where it partly stepped on the right lane and then quickly recovered. I wanted to amend that behaviour and recorded additional recovery of this turn.

After combining forward/backward/challenging curve data, the model cleared the track without the problem. However my driving was quite bad in the backward run and the model inherited nervous angle corrections while driving, which is illustrated below:

Model after the first epoch is preseted in the video format in this folder under 'temp_1epoch.mp4'

Model after the second epoch is preseted in the video format in this folder under 'temp_2epoch.mp4'

Model after the third epoch is preseted in the video format in this folder under 'temp_3epoch.mp4'

Model performance was captured at each epoch and can be illustrated below for the training and validation datasets:

![alt text][image1]


#### 2. Final Model Architecture

The model consists of the convolutional layers with varying filter sizes, ELU activation layers, fully connected layers and the dropout layers with varying probability levels. The data is normalized in the model using a Keras lambda layer

Model architecture illustrated in the flow chart:

![alt text][image3]

The model architecture with dimensions as the extract from model.summary() is provided below:

![alt text][image2]


#### 3. Creation of the Training Set & Training Process

To properly cover the driving situations, I recorded:
- full forward pass through the track
- full backward pass through the track
- recovery of the trickier turn

All three camera views were used with the expertly corrected steering angle for the left and right camera.

Every time the sample was generated, the data was schuffled so that the model doesn't learn the sequences. The training/validation split was performed in the 80%/20% proportion.

The generator performs data augmentation by:
- flipping the image and correcting the steering angle
- adding left/right camera images with the adjusted steering angle

The following picture illustrates the left / center / right camera image:

![alt text][image6]

The following picture illustrates the view after flippping the center image:

![alt text][image5]

The following picture illustrates the view after cropping the center image:

![alt text][image4]

The total number of frames entering the augmentation process and sampling is 2743.

