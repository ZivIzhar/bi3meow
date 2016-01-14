import copy
import pickle
from random import shuffle
import random
import math
from sklearn.tree import DecisionTreeClassifier

from get_features import get_ads_features
from noise import split_to_folds, get_noisy_folds


def get_reduced_data(p,data):
    b = data
    shuffle(b)
    return b[:(math.ceil(len(data)*p))]


def get_reduced_features(p,data):
    newData=[]
    idxs = len(data[0])
    numberOfFeatures = math.ceil((idxs-1)*p)
    featureList = list(range(0, numberOfFeatures))
    shuffle(featureList)
    featureList=featureList[:numberOfFeatures]
    for sample in data:
        newItem=[sample[index] for index in featureList]
        newItem.append(sample[idxs-1])
        newData.append(newItem)
    return newData , featureList


def getTree(number, letter, data):
    newData=[]
    res=[]
    finalData=[]
    featureList=[i for i in range(len(data[0])-1)]
    if number == '1':
        tree=DecisionTreeClassifier(criterion="entropy",splitter="best",min_samples_split=2)
    elif number == '2':
        tree=DecisionTreeClassifier(criterion="entropy",splitter="random",min_samples_split=2)
    else:
        tree=DecisionTreeClassifier(criterion="entropy",splitter="random",min_samples_split=2)
    if letter == 'a':
        newData = get_reduced_data(0.67, data)
    else:
        newData , featureList = get_reduced_features(0.67, data)
    for sample in newData:
        finalData.append(sample[:-1])
        res.append(sample[-1])
    tree = tree.fit(finalData,res)
    return tree , featureList


def check():
    featureList = get_ads_features(313542516, 208346379)
    file = open("ad.data", 'r')
    data = []
    results = []
    for line in file.readlines():
        out = line.split(",")
        item = [out[index] for index in featureList]
        item.append(str(0 if "nonad." in out[-1] else 1))
        data.append(item)
    file.close()
    noisyfolds, folds = get_noisy_folds(data)
    output = open("folds.pkl",'wb')
    output1 = open("noisyfolds.pkl",'wb')
    pickle.dump(folds,output)
    pickle.dump(noisyfolds,output1)
    output.close()
    output1.close()
    sumacc=0
    for i in range(0,10):
        train=[]
        for j in range(0,10):
            if i != j:
                train.extend(noisyfolds[j])
        trainFinal=[]
        results=[]
        for sample in train:
            trainFinal.append(sample[:-1])
            results.append(sample[-1])
        tree=DecisionTreeClassifier(criterion="entropy",splitter="best",min_samples_split=4)
        tree=tree.fit(trainFinal,results)
        predictSamples=[]
        predictResults=[]
        predictSamples = [folds[i][index][:-1] for index in range(len(folds[i]))]
        predictResults = [folds[i][index][-1] for index in range(len(folds[i]))]
        sumacc+=tree.score(predictSamples,predictResults)
    return sumacc/10


def createTrees(size,type,folds,testWith):
    trees=[]
    alldata=[]
    for i in range(len(folds)):
        alldata.extend(folds[i])
    for i in range(size):
        tree, features= getTree(type[1],type[0],alldata)
        trees.append(tree)
    sum=0
    predictSamples=[]
    predictResults=[]
    if(type[0] == 'b'):
        for x in range(len(testWith)):
            predictSamples.append([testWith[x][index] for index in features])
            predictResults.append(testWith[x][-1])
    else:
        predictSamples = [testWith[index][:-1] for index in range(len(testWith))]
        predictResults = [testWith[index][-1] for index in range(len(testWith))]
    predicts=[]
    for tree in trees:
        predicts.append(tree.predict(predictSamples))
    finalPrediction=[]
    for i in range(len(predictResults)):
        predSum=0
        for j in range(len(trees)):
            predSum+=int(predicts[j][i])
        if predSum*2 > size:
            finalPrediction.append(1)
        else:
            finalPrediction.append(0)
    sum=0
    for i in range(len(predictResults)):
        if int(predictResults[i]) == finalPrediction[i]:
            sum+=1
    sum/=len(predictResults)
    return sum

if __name__ == "__main__":
    pkl_folds = open('folds.pkl', 'rb')
    pkl_noisyfolds = open('noisyfolds.pkl', 'rb')
    folds = pickle.load(pkl_folds)
    noisyfolds = pickle.load(pkl_noisyfolds)
    size = [11,21,31,41,51,61,71,81,91,101]
    type = ["a1","a2","a3","b1","b2","b3"]