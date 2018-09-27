# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 05:23:51 2017

@author: rrenv
"""

from __future__ import division
import glob
import scipy.stats
import numpy as np
import matplotlib
import copy
import random 
import math

#
dictHeadPose = {0:"X Euler Angle", 1: "Y Euler Angle", 2: "Z Euler Angle"}
dictAuTranslation = {0 : "AU01", 1 : "AU02", 2 :"AU04", 3:"AU05", 4:"AU06",5:"AU07",6:"AU09",7:"AU10",8:"AU12",9:"AU14",10:"AU15",
                     11:"AU17",12:"AU20",13:"AU23",14:"AU25",15:"AU26",16:"AU28",17:"AU45"}
                     
dictTraitTranslation = {0 : "Openness", 1 : "Conscientiousness", 2 : "Extraversion", 3 : "Agreeableness", 4 : "Neuroticism"}

photoData = {}

#photoData[filename][0] is traits
#photoData[filename][1] is headPose values
#photoData[filename][2] is auIntensities
#photoData[filename][3] is auActivations

def getPhotoData():
    traitFile = open("C:/Users/rrenv/Desktop/Level4Project/averageRatings.csv", "r")
    for line in traitFile:
        line = line[:-1]
        values = line.split(",")
        traitValues = []
        if (values[0][:12] not in photoData):
            photoData[values[0][:12]] = []
            for trait in values[1:]:
                traitValues.append(float(trait))
            photoData[values[0][:12]].append(traitValues)
    traitFile.close()
    print len(photoData)
    filesToGoThrough = glob.glob("C:/Users/rrenv/Desktop/Level4Project/OpenFace_0.2_win_x64/data/outputFiles/*.pts")
    for file in filesToGoThrough:
        auIntensities = []
        auActivations = []
        headPose = []
        filename = file.split("/")[-1][12:][:12]
        currentFile = open(file, "r")
        listOfLines = currentFile.readlines()
        for i in range(0,3):
            value=listOfLines[74].split(" ")[i]
            if (i == 2):
                headPose.append(float(value[0:-1]))
            else:
                headPose.append(float(value))
        photoData[filename].append(headPose)
        for i in range(0,17):
            value=listOfLines[82+i].split(" ")[1]
            auIntensities.append(float(value[0:-1]))
        photoData[filename].append(auIntensities)
        for i in range(0,18):
            value=listOfLines[102+i].split(" ")[1]
            auActivations.append(float(value))
        photoData[filename].append(auActivations)
        currentFile.close()

# X Euler angle is looking up or down,
# Y is looking to the left or right,
# Z is having your head tilted to either side
def getHeadPose(vector, trait):
    vectorValues = [] #Will be a list of either x, y, or z euler angles
    personalityRankings=[]
    minimum = 0.0
    maximum = 0.0
    minFile = ""
    maxFile = ""
    filesToGoThrough = glob.glob("C:/Users/rrenv/Desktop/Level4Project/OpenFace_0.2_win_x64/data/outputFiles/*.pts")
    for file in filesToGoThrough:
        filename = file.split("/")[-1][12:]
        currentFile = open(file, "r")
        listOfLines = currentFile.readlines()
        value=listOfLines[74].split(" ")[vector]
        if (vector == 2):
            vectorValues.append(float(value[0:-1]))
        else:
            vectorValues.append(float(value))
        if (float(value) > maximum):
            maximum = float(value)
            maxFile = filename
        if (float(value) < minimum):
            minimum = float(value)
            minFile = filename
        personalityRankings.append(getPersonalityRanks(trait, filename))
    currentFile.close()
    #print "max value: ", maximum, " min value: ", minimum, " maxFile: ", maxFile, " minFile: ", minFile
    return vectorValues, personalityRankings
    
def getAuActivations(auvalue, trait):
    auBinaries = []
    roundedRankings = []
    filesToGoThrough = glob.glob("C:/Users/rrenv/Desktop/Level4Project/OpenFace_0.2_win_x64/data/outputFiles/*.pts")
    for file in filesToGoThrough:
        filename = file.split("/")[-1][12:]
        currentFile = open(file, "r")
        listOfLines = currentFile.readlines()
        value=int(listOfLines[102+auvalue].split(" ")[1])
        auBinaries.append(value)
        roundedRankings.append(getRoundedRanks(trait, filename))
        currentFile.close()
    return auBinaries, roundedRankings
                      
def getAuIntensities(auvalue, trait): #au
    auIntensities = []
    personalityRankings = []
    filesToGoThrough = glob.glob("C:/Users/rrenv/Desktop/Level4Project/OpenFace_0.2_win_x64/data/outputFiles/*.pts")
    for file in filesToGoThrough:
        filename = file.split("/")[-1][12:]
        currentFile = open(file, "r")
        listOfLines = currentFile.readlines()
        value=listOfLines[82+auvalue].split(" ")[1]
        auIntensities.append(float(value[0:-1]))
        personalityRankings.append(getPersonalityRanks(trait, filename))
        currentFile.close()
    return auIntensities, personalityRankings
    
def getPersonalityRanks(traitNumber, fileName):
    file = open("C:/Users/rrenv/Desktop/Level4Project/averageRatings.csv", "r")
    for line in file:
        values = line.split(",")
        if (values[0] == fileName[0:-10]):
            return float(values[traitNumber+1])
            
def getRoundedRanks(traitNumber, fileName):
    file = open("C:/Users/rrenv/Desktop/Level4Project/averageRatings.csv", "r")
    for line in file:
        values = line.split(",")
        if (values[0] == fileName[0:-10]):
            value = float(values[traitNumber+1])
            return round(value)
    

def getExpectedValues(au, trait):
    total=0	
    auActive=0
    auNonActive=0
    numberOfMinusThree=0
    numberOfMinusTwo=0	
    numberOfMinusOne=0
    numberOfZero=0
    numberOfOne=0
    numberOfTwo=0
    numberOfThree=0
    numberOfFour=0
    for x in range(len(au)):
        if au[x] == 0:
            auNonActive = auNonActive + 1
            total += 1
            #print total
        if au[x] == 1:
            auActive += 1
            total+=1
            #print total
        if(trait[x]== -3):
            numberOfMinusThree += 1
        if(trait[x]==-2):
            numberOfMinusTwo += 1
        if(trait[x] == -1):
            numberOfMinusOne+=1
        if(trait[x] == 0):
            numberOfZero+=1
        if(trait[x] == 1):
            numberOfOne+=1
        if(trait[x] ==2):
            numberOfTwo+=1
        if(trait[x] ==3):
            numberOfThree+=1
        if(trait[x] ==3):
            numberOfFour+=1
    expectedZeroToMinusThree = float(numberOfMinusThree*(auNonActive/total))
    expectedOneToMinusThree= numberOfMinusThree*(auActive/total)
    expectedZeroToMinusTwo = numberOfMinusTwo*(auNonActive/total)
    expectedOneToMinusTwo = numberOfMinusTwo*(auActive/total)
    expectedZeroToMinusOne = numberOfMinusOne*(auNonActive/total)
    expectedOneToMinusOne = numberOfMinusOne*(auActive/total)
    expectedZeroToZero = numberOfZero*(auNonActive/total)
    expectedOneToZero = numberOfZero*(auActive/total)
    expectedZeroToOne = numberOfOne*(auNonActive/total)
    expectedOneToOne = numberOfOne*(auActive/total)
    expectedZeroToTwo = numberOfTwo*(auNonActive/total)
    expectedOneToTwo = numberOfTwo*(auActive/total)	
    expectedZeroToThree = numberOfThree*(auNonActive/total)
    expectedOneToThree = numberOfThree*(auActive/total)		
    expectedZeroToFour = numberOfFour*(auNonActive/total)
    expectedOneToFour = numberOfFour*(auActive/total)
    expectedValues = [expectedZeroToMinusThree, expectedOneToMinusThree, expectedZeroToMinusTwo, expectedOneToMinusTwo, expectedZeroToMinusOne,
								expectedOneToMinusOne, expectedZeroToZero, expectedOneToZero, expectedZeroToOne, expectedOneToOne, expectedZeroToTwo, expectedOneToTwo,
									expectedZeroToThree, expectedOneToThree, expectedZeroToFour, expectedOneToFour]
    return expectedValues
    
def getObservedValues(au, trait):
    numberOfZeroToMinusThree=0
    numberOfOneToMinusThree=0
    numberOfZeroToMinusTwo=0
    numberOfOneToMinusTwo=0
    numberOfZeroToMinusOne=0
    numberOfOneToMinusOne=0
    numberOfZeroToZero=0
    numberOfOneToZero=0
    numberOfZeroToOne=0
    numberOfOneToOne=0
    numberOfZeroToTwo=0
    numberOfOneToTwo=0
    numberOfZeroToThree=0
    numberOfOneToThree=0
    numberOfZeroToFour=0
    numberOfOneToFour=0
    for x in range(len(au)):
        if au[x] == 0:
            if trait[x] == -3:
                numberOfZeroToMinusThree += 1
            if trait[x] == -2:
                numberOfZeroToMinusTwo += 1
            if trait[x] == -1:
                numberOfZeroToMinusOne +=1 
            if trait[x] == 0:
                numberOfZeroToZero += 1
            if trait[x] == 1:
                numberOfZeroToOne += 1
            if trait[x] == 2:
                numberOfZeroToTwo += 1
            if trait[x] == 3:
                numberOfZeroToThree += 1
            if trait[x] == 4:
                numberOfZeroToFour += 1
        elif au[x] == 1:
            if trait[x] == -3:
                numberOfOneToMinusThree += 1
            if trait[x] == -2:
                numberOfOneToMinusTwo += 1
            if trait[x] == -1:
                numberOfOneToMinusOne += 1
            if trait[x] == 0:
                numberOfOneToZero += 1
            if trait[x] == 1:
                numberOfOneToOne += 1
            if trait[x] == 2:
                numberOfOneToTwo += 1
            if trait[x] == 3:
                numberOfOneToThree += 1
            if trait[x] == 4:
                numberOfOneToFour += 1
    return [numberOfZeroToMinusThree, numberOfOneToMinusThree, numberOfZeroToMinusTwo, numberOfOneToMinusTwo,
            numberOfZeroToMinusOne, numberOfOneToMinusOne, numberOfZeroToZero, numberOfOneToZero, numberOfZeroToOne,
            numberOfOneToOne, numberOfZeroToTwo, numberOfOneToTwo, numberOfZeroToThree, numberOfOneToThree,
            numberOfZeroToFour, numberOfOneToFour]
            

def getResults():
    results = open('C:/Users/rrenv/Desktop/Level4Project/Results/allResults.txt', 'w')
    results.write("ChiSquared tests for AU Activations and Personality Traits: \n")
    print "writing ChiSquared results..."
    for i in range (0,18):
        for j in range(0,5):
            auBinaries, roundedRanks = getAuActivations(i,j)
            expectedValues = getExpectedValues(auBinaries, roundedRanks)
            observedValues = getObservedValues(auBinaries, roundedRanks)
            degOfFreedom = 6
            offset=0
            for x in range(0, len(expectedValues)-(1 + offset)):
                #       print x, " : ", expectedValues[x - offset]
                if (expectedValues[x-offset]==0.0):
                    #            print "gooooooteeeeeem: ", expectedValues[x - offset]
                    expectedValues.remove(expectedValues[x - offset])
                    observedValues.remove(observedValues[x - offset])
                    degOfFreedom -= 1
                    offset+=1
            chisquare, pvalue = scipy.stats.chisquare(observedValues, expectedValues, degOfFreedom)
            if ((abs(chisquare) >= 0.0)):
            #chi > 0.3?
            # and (pvalue <= 0.0005555555)):
                results.write(str(dictAuTranslation[i]) + " and " + str(dictTraitTranslation[j]) + ": \n")
                results.write("ChiSquare:" + str(chisquare) + " with " + str((1-pvalue)*100) + "% confidence \n")
                results.write("\n")
                print "..."
    results.write("\n")
    results.write("\n")
    results.write("Spearman Ranked tests for AUIntensities and Personality Trait attributions: \n")       
    print "writing AuIntensity results"
    for i in range (0,17):
        for j in range(0,5):
            auIntensities, traitRankings = getAuIntensities(i,j)
       #print scipy.stats.spearmanr(auIntensities, traitRankings)
            rho, pvalue = scipy.stats.spearmanr(auIntensities, traitRankings)
            if ((pvalue <= 1.00058823529)):
            #pvalue < 0.00058823529 and rho < 0.3?
                results.write(str(dictAuTranslation[i]) + " and " + str(dictTraitTranslation[j]) + ":\n")
                results.write("Rho:" + str(rho) + " with " + str((1-pvalue)*100) + "% confidence\n")
                results.write("\n")
                print "..."
            #print "Spearman value rating for : ", dictAuTranslation[i], " and ", dictTraitTranslation[j], " is : ", rho
            #print "relation between ", dictAuTranslation[i], " and ", dictTraitTranslation[j], " is very unlikely to be due to chance"
            #print "But could be due to chance"
        results.write("\n")
        results.write("\n")
        results.write("Spearman Ranked tests for HeadPose and Personality Trait attributions:\n")
        print "writing head pose results"
        for i in range(0,3):
            for j in range(0,5):
                vectorValues, personalityRankings = getHeadPose(i, j)
                rho, pvalue = scipy.stats.spearmanr(vectorValues, personalityRankings)
                if (pvalue <= (1.00333333333)):
            #pvalue <= 0.00333333333
                    results.write(str(dictHeadPose[i]) + " and " + str(dictTraitTranslation[j]) + ": \n")
                    results.write("Rho:" + str(rho) +  " with " + str((1-pvalue)*100) + "% confidence\n")     
                    results.write("\n")
                    print "..."
        results.close()
        print "done with all results..."

        print "now filtering out the significant ones..."

    sigResults = open('C:/Users/rrenv/Desktop/Level4Project/Results/SignificantResults.txt', 'w')
    sigResults.write("ChiSquared tests for AU Activations and Personality Traits: \n")
    print "writing ChiSquared results..."
    for i in range (0,18):
        for j in range(0,5):
            auBinaries, roundedRanks = getAuActivations(i,j)
            expectedValues = getExpectedValues(auBinaries, roundedRanks)
            observedValues = getObservedValues(auBinaries, roundedRanks)
            degOfFreedom = 6
            offset=0
            for x in range(0, len(expectedValues)-(1 + offset)):
                if (expectedValues[x-offset]==0.0):
                    expectedValues.remove(expectedValues[x - offset])
                    observedValues.remove(observedValues[x - offset])
                    degOfFreedom -= 1
                    offset+=1
                chisquare, pvalue = scipy.stats.chisquare(observedValues, expectedValues, degOfFreedom)
                if ((abs(chisquare) >= 0.3) and (pvalue <= 0.00055555555)):
            #chi > 0.3?
            # and (pvalue <= 0.0005555555)):
                    sigResults.write(str(dictAuTranslation[i]) + " and " + str(dictTraitTranslation[j]) + ": \n")
                    sigResults.write("ChiSquare:" + str(chisquare) + " with " + str((1-pvalue)*100) + "% confidence \n")
                    sigResults.write("\n")
            print "..."
            #print "ChiSquare values rating for : ", dictAuTranslation[i], " and ", dictTraitTranslation[j], " is ", chisquare, "and very unlikely due to chance"
#       
    sigResults.write("\n")
    sigResults.write("\n")
    sigResults.write("Spearman Ranked tests for AUIntensities and Personality Trait attributions: \n")       
    print "writing AuIntensity results"
    for i in range (0,17):
        for j in range(0,5):
            auIntensities, traitRankings = getAuIntensities(i,j)
       #print scipy.stats.spearmanr(auIntensities, traitRankings)
            rho, pvalue = scipy.stats.spearmanr(auIntensities, traitRankings)
            if ((pvalue <= 0.00058823529) and (rho >= 0.07)):
            #pvalue < 0.00058823529 and rho < 0.3?
                sigResults.write(str(dictAuTranslation[i]) + " and " + str(dictTraitTranslation[j]) + ":\n")
                sigResults.write("Rho: " + str(rho) + " with " + str((1-pvalue)*100) + "% confidence\n")
                sigResults.write("\n")
                print "..."
            #print "Spearman value rating for : ", dictAuTranslation[i], " and ", dictTraitTranslation[j], " is : ", rho
            #print "relation between ", dictAuTranslation[i], " and ", dictTraitTranslation[j], " is very unlikely to be due to chance"
            #print "But could be due to chance"            sigResults.write("\n")
    sigResults.write("\n")
    sigResults.write("Spearman Ranked tests for HeadPose and Personality Trait attributions:\n")
    print "writing head pose results"
    for i in range(0,3):
        for j in range(0,5):
            vectorValues, personalityRankings = getHeadPose(i, j)
            rho, pvalue = scipy.stats.spearmanr(vectorValues, personalityRankings)
            if (pvalue <= (0.00333333333)):
            #pvalue <= 0.00333333333
                sigResults.write(str(dictHeadPose[i]) + " and " + str(dictTraitTranslation[j]) + ": \n")
                sigResults.write("Rho: " + str(rho) +  " with " + str((1-pvalue)*100) + "% confidence\n")     
                sigResults.write("\n")
                print "..."
    sigResults.close()
    print "done."
    
#-----------------------------------------------------------------------------------------------------------------#
#-------------------------------OUTPUT AND BUBBLE CHART MAKING CODE-----------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

# Write Results to spreadsheet files where each sell is a tuple such that (statistical significance value, pvalue)
# BUT remember to have pvalue simply written as >.05 or >.01, having taking into account the Bonferroni correction,
# so really, it should be >0.00055555555 and >.000111111111

import plotly.plotly as py
import plotly.graph_objs as go

trace0 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    text=['A</br>size: 40</br>default', 'B</br>size: 60</br>default', 'C</br>size: 80</br>default', 'D</br>size: 100</br>default'],
    mode='markers',
    marker=dict(
        size=[400, 600, 800, 1000],
        sizemode='area',
    )
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[14, 15, 16, 17],
    text=['A</br>size: 40</br>sixeref: 0.2', 'B</br>size: 60</br>sixeref: 0.2', 'C</br>size: 80</br>sixeref: 0.2', 'D</br>size: 100</br>sixeref: 0.2'],
    mode='markers',
    marker=dict(
        size=[400, 600, 800, 1000],
        sizeref=2,
        sizemode='area',
    )
)
trace2 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[20, 21, 22, 23],
    text=['A</br>size: 40</br>sixeref: 2', 'B</br>size: 60</br>sixeref: 2', 'C</br>size: 80</br>sixeref: 2', 'D</br>size: 100</br>sixeref: 2'],
    mode='markers',
    marker=dict(
        size=[400, 600, 800, 1000],
        sizeref=0.2,
        sizemode='area',
    )
)

data = [trace0, trace1, trace2]
py.iplot(data, filename='bubblechart-size-ref')

#-----------------------------------------------------------------------------------------------------------------#   
#------------------------------------------CLASSIFIER CODE--------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

def getRoundedPhotoData():
    traitFile = open("C:/Users/rrenv/Desktop/Level4Project/averageRatings.csv", "r")
    for line in traitFile:
        line = line[:-1]
        values = line.split(",")
        traitValues = []
        if (values[0][:12] not in photoData):
            photoData[values[0][:12]] = []
            for trait in values[1:]:
                traitValues.append(round(float(trait)))
            photoData[values[0][:12]].append(traitValues)
    traitFile.close()
    #print len(photoData)
    filesToGoThrough = glob.glob("C:/Users/rrenv/Desktop/Level4Project/OpenFace_0.2_win_x64/data/outputFiles/*.pts")
    for file in filesToGoThrough:
        auIntensities = []
        auActivations = []
        headPose = []
        filename = file.split("/")[-1][12:][:12]
        currentFile = open(file, "r")
        listOfLines = currentFile.readlines()
        for i in range(0,3):
            value=listOfLines[74].split(" ")[i]
            if (i == 2):
                headPose.append(float(value[0:-1]))
            else:
                headPose.append(float(value))
        photoData[filename].append(headPose)
        for i in range(0,17):
            value=listOfLines[82+i].split(" ")[1]
            auIntensities.append(float(value[0:-1]))
        photoData[filename].append(auIntensities)
        for i in range(0,18):
            value=listOfLines[102+i].split(" ")[1]
            auActivations.append(float(value))
        photoData[filename].append(auActivations)
        currentFile.close()
        
#Split into Training and Testing

def splitData(data, ratio):
    trainingSize = int(len(data) * ratio)
    trainSet = []
    copyOfData = copy.copy(data)
    while (len(trainSet) < trainingSize):
        trainSet.append(copyOfData.pop(random.choice(copyOfData.keys())))
    return [trainSet, copyOfData]

#Separate Data by Class

def assignOnScale(dataset, variable): #Variable will mean which of the 5 personality traits you are trying to classify
    assigned = {}
    for i in range(len(dataset)):
        #print "working with: ", dataset[i]
        vector = dataset[i][0][variable]
        if (vector not in assigned):
            assigned[vector] = []
        for j in range(1,len(dataset[i])):
            assigned[vector].append(dataset[i][j])
    return assigned

#Get Mean for each Attribute

def getMean(feature):
    return (sum(feature)/len(feature))

#Get Standard Deviation

def getStdDev(feature):
    mean = getMean(feature)
    variance = sum([pow(x-mean,2) for x in feature])/float(len(feature)-1)
    return np.sqrt(variance)

#Summarise Dataset

def getAllFeatures(dataset, parameter):
    features = {}
    for datapoint in dataset:
        for x in range(1, len(datapoint)): #since [0] would be the class
            if x in features:
                features[x].append(datapoint[x])
            else:
                features[x] = []
                features[x].append(datapoint[x])
    return features

    
def summariseFeatures(dataset):
    features = getAllFeatures(dataset)
    summaries = {}
    for x in range(1,len(features)):
        mean = getMean(features[x])
        stdDev = getStdDev(features[x])
        summaries[x] = []
        summaries[x].append(mean)
        summaries[x].append(stdDev)
    return summaries

#Summarise Attributes by Class

def summariseClasses(dataset):
    assigned = assignOnScale(dataset)
    summaries = {}
    for classValue, instances in assigned.iteritems():
        summaries[classValue] = summariseFeatures(instances)
    return summaries

#Make Predictions based on Probabilities

#Gaussian Probablilty (probability of a given value given mean and standard deviation)

def findprobability(value, mean, stdev): # value is what we're looking for the probability of
    exponent = math.exp(-(math.pow(value-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

#Apply to classes in our data

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(1, len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            if (stdev != 0):
                probabilities[classValue] *= findprobability(x, mean, stdev)
    return probabilities
    
allData = getRoundedPhotoData()
trainSet, testSet = splitData(photoData, 0.66)
#print trainSetfeatures = assignOnScale(trainSet, 2)
#print len(trainSet)
#for feature, measurements in features.iteritems():
#    print feature, ": ", (len(measurements))/3