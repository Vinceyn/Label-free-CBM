# Questions

In this file, I will put all of the interrogations I have regarding the code 

1. In `train_cbm.py`, in the data preprocessing before the training of the path between CBM and target around line 200:
The authors preprocess the training data by going through the projection layer and normalizing the data. Then, they pair it with the targets they have in the indextensordataset. 

2. In `train_cbm.py`, when computing `clip_features` and `val_clip_features`, 
Both variables go through a different filtering, so I don't understand how they can both be used in the same model
