# VT-ADL with applied pre trained weights? 🤔



## First we download dataset(I mean, clone this repo first of course)
### 1. curl -L -o datasets https://www.kaggle.com/api/v1/datasets/download/ipythonx/mvtec-ad
### 2. save this into a folder named datasets
#### you should end up with this type of structure - datasets/mvtec/(all the sub folders, bottle, hazelnut etc)

## Secondly, install libraries
### pip install -r requirements.txt

## Then just run everything in the train.ipynb.








# Notes - The repo is very dirty(not cleaned), so please bare with me.



progress updates for emerging technology 

1. replaced the transformer encoder with pretrained encoder from microsoft(/Users/dania/code/EmergingTechnologies/VT-ADL/datasets/mvtec)
2. trained the model for 60 epochs(2 hours)
3. the loss has not changed, need to update backwards()