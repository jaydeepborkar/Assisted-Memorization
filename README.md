This repository contains code to reproduce results from our paper: 

**Privacy Ripple Effects from Adding or Removing Personal Information in Language Model Training**  
Jaydeep Borkar, Matthew Jagielski, Katherine Lee, Niloofar Mireshghallah, David A. Smith, and Christopher A. Choquette-Choo

## Software
- Required for training and inference: transformers, datasets, accelerate, and torch. We use transformers v4.44.0, datasets v2.14.7, accelerate v0.29.2, torch v2.2.2, and python v3.9.12.

## Training
**Continuous Training Setup**: To train a model and checkpoint it every 10% of training, run:  
  ```python training.py continue_train``` 

  This will save the checkpoints in the ```models``` directory and data seen during training in ```data``` directory. 

 **Retraining Setup**: To train all of our ten models for this setup, run:  
 ```python training.py retrain``` 

 This will save all the models in ```models``` directory. 
