# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)



[sample_image_with_label]: ./output_images/sample_image_with_label.png "Sample Image"
[class_count_training_set]: ./output_images/class_count_training_set.png "Class Count Training Set"
[class_count_validation_set]: ./output_images/class_count_validation_set.png "Class Count Validation Set"
[class_count_test_set]: ./output_images/class_count_test_set.png "Class Count Test Set"
[training_accuracy]: ./output_images/training_accuracy.png "Training Accuracy"
[image1]: ./new_img/001.jpg "Traffic Sign 1"
[image2]: ./new_img/002.jpg "Traffic Sign 2"
[image3]: ./new_img/003.jpg "Traffic Sign 3"
[image4]: ./new_img/004.jpg "Traffic Sign 4"
[image5]: ./new_img/005.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kurokis/SDCND-P3-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an example of an image contained in the dataset.

![alt_text][sample_image_with_label]

The charts below are bar charts of class ID counts in training, validation, and test sets, respectively.

![alt text][class_count_training_set]

![alt text][class_count_validation_set]

![alt text][class_count_test_set]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because
the LeNet architecture was originally designed to handle grayscale images,
so I thought it would be a good starting point.

Then I normalized the image because neural networks work best when the dataset has zero mean.
I noticed that some images have high contrast, and others have low contrast.
Therefore I decided to divide the zero-meaned imaged by the standard deviation

To add more data to the the data set, I used the following techniques provided by Pierre Sermanet and Yann LeCun in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

 - Perturb image by 2 pixels
 - Scale image by factor of 0.9 and 1.1
 - Rotae image by 15 degrees

Performing such augmentation increased the size of training dataset to 313191, which is 
9 times more than the original.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten		      	| inputs 5x5x16,  outputs 400 					|
| Fully connected       | inputs 400,  outputs 120 	    				|
| RELU					|												|
| Fully connected       | inputs 120,  outputs 84   					|
| RELU					|												|
| Fully connected       | inputs 84,  outputs 43    					|
| Softmax				|           									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer with following hyperparameters.

|Hyperparameter|Description|
|:---:|:---:|
|Learning rate|0.001|
|Epochs|20|
|Batch size|128|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.934
* test set accuracy of 0.932

<!---
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
-->

<!---
If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
-->

I chose to use the well known LeNet architecture for this project.
I determined this would be a good architecture because traffic sign classification is a very similar task to handwriting recognition, especially in this dataset where the traffic sign occupies most of the area in the image.
For this task, I changed the number of outputs in the final fully connected layer from 10 to 43.
If I were to detect traffic signs from a wide-angle camera, I would need to use some other architecture which can find candidate areas when traffic signs might exist in the image.

However, there were following concerns.
- LeNet uses grayscale images, whereas traffic sign dataset contains RGB images
- Original LeNet architecture successfully classified 10 classes, whereas traffic sign classification task must classify 43 classes

As for the colors, I simply converted images to grayscale and see how well the classifier performs. For handling more classes, I decided to augment the dataset to better capture traffic sign features.


Here is the history of prediction accuracy for training and validation datasets.

![alt_text][training_accuracy]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

* 1st image is an electric road sign of speed limit (100km/h)
* 2nd image is a tilted sign of slippery road
* 3rd image is a tilted sign of bumpy road
* 4th image is a fairly clean image of ahead only sign
* 5th image is a no passing sign with some dirt

Since image size is not 32x32 which the classifier takes as input, I preprocessed them using OpenCV resize function.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (100km/h) | Speed limit (70km/h)   						| 
| Slippery road         | Slippery road 				    			|
| Bumpy road			| Bumpy road									|
| Ahead only	     	| Ahead only					 				|
| No passing        	| No passing            	   					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The classifier was very confident in predicting the labels.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Overall, the classifer was very confident in its predictions, even when its prediction was incorrect.

##### 1st image

![alt text][image1]

For the first image, the model **incorrectly** predicted the label to be "speed limit (70km/h)" with confidence of 100.00%. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%         		| Speed limit (70km/h)						| 
|   0.00%     			| Stop 										|
|   0.00%				| Speed limit (80km/h)						|
|   0.00%	      		| Ahead only								|
|   0.00%			 	| Speed limit (120km/h)      				|

##### 2nd image

![alt text][image2]

For the second image, the model **correctly** predicted the label to be "slippery road" with confidence of 100.00%. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%         		| Slippery road						| 
|   0.00%     			| Dangerous curve to the left 		|
|   0.00%				| Dangerous curve to the right		|
|   0.00%	      		| Speed limit (60km/h)				|
|   0.00%			 	| Children crossing      			|


##### 3rd image

![alt text][image3]

For the third image, the model **correctly** predicted the label to be "bumpy road" with confidence of 100.00%. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%         		| Bumpy road				| 
|   0.00%     			| Speed limit (20km/h) 		|
|   0.00%				| Speed limit (30km/h)		|
|   0.00%	      		| Speed limit (50km/h)		|
|   0.00%			 	| Speed limit (60km/h)     	|

##### 4th image

![alt text][image4]

For the fourth image, the model **correctly** predicted the label to be "ahead only" with confidence of 100.00%. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%         		| Ahead only				| 
|   0.00%     			| Go straight or left 		|
|   0.00%				| Go straight or right		|
|   0.00%	      		| Road work					|
|   0.00%			 	| Turn left ahead      		|

##### 5th image

![alt text][image5]

For the fifth image, the model **correctly** predicted the label to be "no passing" with confidence of 100.00%. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%         		| No passing									| 
|   0.00%     			| Vehicles over 3.5 metric tons prohibited 		|
|   0.00%				| End of no passing								|
|   0.00%	      		| No passing for vehicles over 3.5 metric tons	|
|   0.00%			 	| Ahead only     								|

<!---
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
-->

