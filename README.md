# EventPriming_RVT
Priming events by background filtering with bbox cues from RGB inference

Dataset used : MOD-DSEC https://github.com/ZZY-Zhou/RENet?tab=readme-ov-file

Base Model used : RVT   https://github.com/uzh-rpg/RVT

To use the code in this repository, clone the RVT model and download the MOD-DSEC dataset.

**Objective** :
Data preprocessing from MOD-DSEC dataset into RVT compatible dataset.
The RGB part of the dataset is used in a RGB-based object detection model for RGB-based prediction.
the RGB inference is used in the Priming process to enhance the event-representations
RVT model is then applied to this primed events for improved detection performance.
Demonstrating the potential in 'Enhancing Precision in Event-Based Object Detection by Integrating Insights from Low-Frequency Frames'

![image](https://github.com/lkxv333/EventPriming_RVT/assets/83161265/b9574358-811b-419a-a849-82f0cb459ab9)

**Methodology**

**1. Preparing RGB inference**

This process require a choice of your preferred RGB based object detection model.
Apply this to get the bounding box and classification of the RGB frames of the MOD-DSEC dataset.


**2. Preprocessing**

Use the 'rectify_preprocess.py' to prepare the MOD-DSEC dataset by transforming into a form that the RVT model expects.
This step consist of three main components
   - resizing of events frame to match the RVT input size
   - event rectification for better alignment
   - creating labels, objframe timesteps and other representations required for RVT.
Make sure to check the paths are assigned correctly.


**3. Priming**

Use the 'prime_autolabel.py' code to prime with a frequency of your choice.
The preprocessed sequences from preprocessing step should be used in this part.


**4. Evaluation**

Use the RVT model evaluation code on the primed events.
The RVT model do not produce sequence by sequency detection output. For visualisation of the predictions, use the 'coco_eval.py' from this repository
This version is adjusted one that creates a detection.py file which contains the predictions from the RVT model when RVT evaluation.py is ran.
Also additional metrics by each class is added in this version.

![image](https://github.com/lkxv333/EventPriming_RVT/assets/83161265/908d70d2-51db-4c21-9b25-5308e4411011)



**5. Visualisation**

Use choice_visual.py to create a visualisation video for a selected sequence.
You may adjust the parameters for which to be included in the video among rgb, events, detection and label.


**Caution**

For best outcome, classification of the RGB inference model and corresponding RVT model should have the same class definitions to maintain class consistency.
The divergence stems from a fundamental difference in how these systems have been trained to recognize and categorize object. And  This results in underestimating the mAP score. 
The RVT model has been trained to take a broader approach, classifying all types of vehicles — be it vans, trucks, or cars — under a singular 'car' category,  while the Grounding DINO model(RGB inference) narrows down the 'car' label to standard passenger cars only
In addition, the RVT model does not recognize cyclists, which leads to a significant classification gap as the frame-based methods tags cyclists as pedestrians as well.

![image](https://github.com/lkxv333/EventPriming_RVT/assets/83161265/92fec6fe-e3a8-46d6-a4df-59066700daff)



Be sure to manage the directories correctly as the preprocessing and priming need the correct form of the dataset.

