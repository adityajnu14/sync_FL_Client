import pandas as pd
import numpy as np
import json
import base64
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import time



# Returning 0 indicates successfull execution

'''
    This function stores(send) latest model to respective folder.
    It works as model transfer machanism for various participating devices. 
'''

def saveLearntMetrice(file_name,score):

    with open(file_name,'r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))

def FetchClientResults():
    print("-------------Inside fetch client---------")
    return "result from client"

# The function is used for training local model based on private data
def Train(modelString):
    

    print("Starting training...")

    continousTrainingBatchSize = 60
    #get latest model from own directory
    with open("data/current_model.h5","wb") as file:
        file.write(base64.b64decode(modelString))
    model = load_model("data/current_model.h5")

    #Reading index to simulate continous learning
    currentIndex = 0
    with open('data/indexFile.txt', "r+") as f:
        fileIndex = json.load(f)
        currentIndex = fileIndex['index']

    print("Current Index is ", currentIndex)

    data = pd.read_csv('data/data.csv')
    
    totalRowCount = data.shape[0]
    nextIndex = currentIndex + continousTrainingBatchSize if currentIndex + continousTrainingBatchSize < totalRowCount else totalRowCount
    X = data.iloc[currentIndex:nextIndex,1:-1].values
    y = data.iloc[currentIndex:nextIndex,-1].values
    y = to_categorical(y)

    #print("Dimension of current data ", X.shape)

    #Updating Index
    if nextIndex == totalRowCount:
        nextIndex = 0
    with open('data/indexFile.txt', "w") as f: 
        index = {'index' : nextIndex}
        f.write(json.dumps(index))


    #Printing aggregated global model metrics
    score = model.evaluate(X, y, verbose=0)
    print("Global model loss : {} Global model accuracy : {}".format(score[0], score[1]))
    
    saveLearntMetrice('data/metrics.txt', score)


    model.fit(X, y, epochs=5, verbose=0)
           
    #Printing loss and accuracy after training 
    score = model.evaluate(X, y, verbose=0)
    print("Local model loss : {} Local model accuracy : {}".format(score[0], score[1]))
    
    saveLearntMetrice('data/localMetrics.txt', score)

    #Save current model 
    model.save('data/model.h5')
    with open('data/model.h5','rb') as file:
        encoded_string = base64.b64encode(file.read())
    
    print("Local training completed")
    print("Extra lines")
    return encoded_string


    
#This function is used fro dataset generation
def InitilizeClients():
    metric = {'accuracy' : [], 'loss' : []}
    index = {'index' : 0}

    with open('data/indexFile.txt', "w") as f:
        f.write(json.dumps(index))
    with open('data/metrics.txt', "w") as f:
        f.write(json.dumps(metric))
    with open('data/localMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

    print("Devices initilization done")
    return 0