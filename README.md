This repository contains code to reproduce results from our paper: 

**Privacy Ripple Effects from Adding or Removing Personal Information in Language Model Training**  
Jaydeep Borkar, Matthew Jagielski, Katherine Lee, Niloofar Mireshghallah, David A. Smith, and Christopher A. Choquette-Choo

## Software
- Required for training and inference: transformers, datasets, accelerate, torch, and tqdm. We use transformers v4.44.0, datasets v2.14.7, accelerate v0.29.2, torch v2.2.2, and python v3.9.12.

## Step 1: Training

Note: Considering the sensitive nature of our datasets, we will make them available on a request basis. If you would like to access our datasets for training, please send an email to borkar.j@northeastern.edu. 
### **Continuous Training Setup**: 

To train a model and checkpoint it every 10% of training, run:  
  ```python training.py continue_train``` 

  This will save the checkpoints in the ```models``` directory and data seen during training in ```data``` directory. Next, you should also run ```python process_data_files.py <folder_path>``` and ```python process_checkpoints.py <folder_path>``` where ```<folder_path>``` is the path to your directory containing data files and checkpoints. This will re-name the data files and checkpoints to more readable structure containing epoch name and training interval. 

 ### **Retraining Setup**: 
 
 To train all of our ten models for this setup, run:  
 ```python training.py retrain``` 

 This will save all the models in ```models``` directory. 

 ## Step 2: Generating Samples 

 First, you will need to download a slice of Common Crawl that we will use to prompt our models. You can do this using ```./commoncrawl.sh```. This will download a WET file ```crawl.wet``` from the December 2024 crawl for you. You can also use the crawl of [your choice](https://commoncrawl.org/get-started). 

 To generate samples, run:  
 ```python extract.py --wet_file crawl.wet --batch_size 100 --num_samples 25000 --max_langth 256```  
 
 You can adjust ```batch_size``` according to your compute. This will save your generations in a txt file. 

 ## Step 3: Evaluating for Memorization 
 To evaluate memorization in the Continuous Training setup and generate a plot, run:  
 ```python taxonomy.py```

 This will also save memorized examples with taxonomy labels in the form of CSV files. 

 For the Retraining Setup, run:  
 ```python pii_add.py``` 
