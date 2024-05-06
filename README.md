# EventPriming_RVT
Priming events by background filtering with bbox cues from RGB inference

Dataset used : MOD-DSEC https://github.com/ZZY-Zhou/RENet?tab=readme-ov-file
Base Model used : RVT   https://github.com/uzh-rpg/RVT

To use the code in this repository, clone the RVT model and download the MOD-DSEC dataset.


1. Preparing RGB inference
This process require a choice of your preferred RGB based object detection model.
Apply this to get the bounding box and classification of the RGB frames of the MOD-DSEC dataset.

2. Preprocessing
Use the 'rectify_preprocess.py' to prepare the MOD-DSEC dataset by transforming into a form that the RVT model expects.
This step consist of three main components
  1. resizing of events frame to match the RVT input size
  2. event rectification for better alignment
  3. creating labels, objframe timesteps and other representations required for RVT.
Make sure to check the paths are assigned correctly.

3. Priming
Use the 'prime_autolabel.py' code to prime with a frequency of your choice.
The preprocessed sequences from preprocessing step should be used in this part.

4. Evaluation
Use the RVT model evaluation code on the primed events.
The RVT model do not produce sequence by sequency detection output. For visualisation of the predictions, use the 'coco_eval.py' from this repository
This version is adjusted one that creates a detection.py file which contains the predictions from the RVT model when RVT evaluation.py is ran.
Also additional metrics by each class is added in this version.

5. Visualisation
Use choice_visual.py to create a visualisation video for a selected sequence.
You may adjust the parameters for which to be included in the video among rgb, events, detection and label.

