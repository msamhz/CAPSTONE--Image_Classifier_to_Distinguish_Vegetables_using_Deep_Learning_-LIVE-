# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Image Classifier to Distinguish Vegetables using Deep Learning (LIVE)

# Background

Googling names and referencing to pictures is easy. But looking at the vegetables and telling the name of it is difficult. Especially tricky ones like Kailan and Chyesim.

A poll was done, and half the participants got the guess wrong just based on image shown, hence further motivates my journey to pursue this project.

## Problem Statement

As a Data scientist enthusiast, I will like to explore on how we can create an image classifier that is able to distinguish difficult vegetables and help give good predictions for users using Live webcam. 
Using transfer learning, I aim to study the architecture behind VGG16/MobileNetV2 and train the model to best fit on my dataset of vegetables. 

**Goal:**

1) To create a model that is accurate in predicting images of vegetables using CNN modelling for mutli-class classification. 
2) Enable classification model into image upload and Live webcam use. 


**Further Goal/Future work**

- Can be used as add on delivery services, whereby instead of figuring out what to search, use the camera instead. 
- Auto-pricing mechanism, whereby price can be tagged by image of the type of vegetable, instead of classifying vegetables manually and putting labels with price on it.
- Essentially deploy MobileNetV2 into phones as it was proven to be low memory consuming as compared to VGG16. 


**Executive Summary** 

Computer vision has been a boon in the industry nowadays, and it has been rapidly growing. Over the course of this project, there were many areas that needed huge attention and were tasked to work independently as most of the in-depth topics were not taught in class. 
Deploying the methods in webcam and troubleshooting the issue with MobileNetV2 initial transfer learning phase. I personally feel that this project, if deployed successfully, will actually help lend a hand to the young adults who does not go to the market often
a chance to better identify vegetables they are unsure off. It could vary with uses such as taking image of vegetable to get the name, and order it online from Shopee for example. If in a wet market dealing with unlabelled vegetables, users may use such app to identify unfamiliar vegetables efficiently. 

Computer vision nowadays rely on deep learning techniques and neural network, hence there is bound to have huge processing capabilities. The details below are libraries used and hardware used for this project. 


`Libraries used`

Tensor Flow Version: 2.1.0
Keras Version: 2.2.4-tf

Python 3.7.13 (default, Mar 28 2022, 08:03:21) [MSC v.1916 64 bit (AMD64)]
Pandas 1.3.4
Scikit-Learn 1.0.2


`Hardware use`

GPU: NVIDIA GeForce RTX 2080 
Disk Drive: Samsung SSD 980 250GB 
RAM: 16GB (In my honest opinion, it was not enough to hold all trained models at one go, hence checkpoints and saving models were crucial for training. 
Webcam: C922 Pro Stream Webcam


**CNN Models Used**
- Base model (2 convolution layers, Flattening, 2 layers of fully connected network and softmax activation function). 
- Transfer learning: VGG 16
- Transfer learning: MobileNetV2

**Key Findings** 

- Putting more images per class, having imbalanced classes for multiclass problem will not have issue with accuracy. Verified with simple modelling, the more we insert data into these classes, the more accurate the model can be. 

![image](https://user-images.githubusercontent.com/98629542/168134291-31333d94-f48e-4d6c-8102-c5f5c9d1393d.png)

- VGG 16 uses high amount of memory space. 
- MobileNetV2 was initially facing Validation loss freeze, but was rectified after retraining the whole model from the first convolution layer, as it was hypothesized that the convolution layer before a series of bottleneck layers might not be good enough to provide sufficient key features into these small tensor dimensions. 

![image](https://user-images.githubusercontent.com/98629542/168134362-d2f7e677-f3d0-405d-b727-87efd41558c7.png) ![image](https://user-images.githubusercontent.com/98629542/168134655-53f891eb-aed8-4e95-8dc5-3bd1ba55c5fa.png)


- Feature map was used to compare between MobileNetV2_pretrained weights model VS MobileNetV2_Unfrozen and retrained model, found that latter has better convolutions, giving more valuable output of features before feeding the information into bottleneck layers. 

![image](https://user-images.githubusercontent.com/98629542/168134750-278dc1f6-3fd9-4944-a1d9-30f92919a436.png)


- Created Live Webcam with two main features 
	- Top 3 predictions of which class (To give users a sense of confidence level of how true the model is predicting the unseen data, ie. having top prediction prob being a Daun_sup is 90% while other two predictions are at 5% each, it shows how confident the model is at segregating the correct class from the other two wrong classes. 
	- Overlayed one image following the top prediction layer. This is to help user compare with the predicted image vs the unseen data, to have a better gauge if the model is actually predicting it accurately aside from just referencing the probability score. 


![image](https://user-images.githubusercontent.com/98629542/168134108-4189f28b-330c-447b-adf0-63f452ddf4cd.png)

**Error Analysis** 

More information can be found in the slides for error analysis for all 3 Models. 

Final production model: `MobileNetV2` 
![image](https://user-images.githubusercontent.com/98629542/168134845-9a0869d9-5b91-45a6-83d6-21ee6cec2a2a.png)
![image](https://user-images.githubusercontent.com/98629542/168134860-c0cccbe3-ca21-4f2e-980b-393211eb6db6.png)


## Conclusions and Recommendations

To have a good image classifier, it is very crucial we do proper data cleaning, including taking the right photos and ensuring only features of the vegetables and do not introduce too much variables that may seem important to the model, for example including too much of the hand while taking the photos. 

In global context, when we are dealing with image gathering through cloud or online spaces, we have to be mindful of filtering away pictures that has incriminating features like the label names it self,ie. taking pictures of cabbages with the background indicating labels: 'Cabbage'. 

Also, retraining models may consist a few iterations. It helps when we better understand the model we pick for transfer learning as not all models are good for our specific use cases, like vegetables. 
We may need look at the pre-trained images used for the model, and understand the features captured. 
We also have to consider the memory usage if we are thinking of using these models in our phones, in this case, MobileNetV2 was a best pick for production model showing high accuracy scores and good prediction without using much memory space. 

Assuming deployment using `Live Video`, we can put more effort into expanding the dataset with cleaner pictures from cloud spaces 
To cater for `wrongly classified images` or `missing classes`;
1) Deploying an option for users to input simple feedback if they don’t think that the top predicted image matches the unseen data
2) Video to autosave and used as input file to learn new data by frames
Zoom into classes that has lower defining probabilities between Top and 2nd predictions.  
(50% - Top prediction, 40% - 2nd prediction) 




