import copy
import pickle
from random import shuffle
import random
import math
from sklearn.tree import DecisionTreeClassifier

from get_features import get_ads_features
from noise import split_to_folds, get_noisy_folds


def get_reduced_data(p,data):
    b = random.sample(data,len(data))
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
        tree=DecisionTreeClassifier(criterion="entropy",splitter="best",min_samples_split=4)
    elif number == '2':
        tree=DecisionTreeClassifier(criterion="entropy",splitter="random",min_samples_split=4)
    else:
        tree=DecisionTreeClassifier(criterion="entropy",splitter="random",min_samples_split=4)
    if letter == 'a':
        print("mitsi")
        newData = get_reduced_data(0.67, data)
    else:
        print("meow")
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


def runTrees(size,type,folds,noisyfolds):
    sumtree=[0 for j in range(size)]
    summacc=0
    for i in range(0,10):
        trees=[]
        features=[]
        train=[]
        for j in range(0,10):
            if i != j:
                train.extend(noisyfolds[j])
        for j in range(size):
            tree ,featureList =getTree(type[1],type[0],train)
            trees.append(tree)
            features.append(featureList)

        for j in range(size):
            predictSamples=[]
            predictResults=[]
            if(type[0] == 'b'):
                for x in range(len(folds[i])):
                    predictSamples.append([folds[i][x][index] for index in features[j]])
                    predictResults.append(folds[i][x][-1])
            else:
                predictSamples = [folds[i][index][:-1] for index in range(len(folds[i]))]
                predictResults = [folds[i][index][-1] for index in range(len(folds[i]))]
            sumtree[j]+=trees[j].score(predictSamples,predictResults)
    for i in range(size):
        sumtree[i] /= 10
    return sumtree


if __name__ == "__main__":
    pkl_folds = open('folds.pkl', 'rb')
    pkl_noisyfolds = open('noisyfolds.pkl', 'rb')
    folds = pickle.load(pkl_folds)
    noisyfolds = pickle.load(pkl_noisyfolds)
    size = [11,21,31,41,51,61,71,81,91,101]
    type = ["a1","a2","a3","b1","b2","b3"]