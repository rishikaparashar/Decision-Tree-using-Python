# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:57:25 2018

@author: Rishika
"""
import pandas as pd
import math

data = pd.read_csv("data/training_set.csv")
testdata = pd.read_csv("data/test_set.csv")
attributes = list(data.drop('Class', axis=1))
target_attr_values = list(data['Class'])

#function to find eliminate duplicate values from dataset
def _unique(seq, return_counts=False, id=None):
    
    found = set()
    if id is None:
        for x in seq:
            found.add(x)
            
    else:
        for x in seq:
            x = id(x)
            if x not in found:
                found.add(x)
    found = list(found)            
    counts = [seq.count(0),seq.count(1)]
    if return_counts:
        return found,counts 
    else:
        return found
      
#function to get the sum of values
def _sum(data):
    sum = 0
    for i in data:
        sum = sum + i
    return sum

#function to find the index value of max count in the data
def _argmaxindex(valuelist):
    count_neg = valuelist.count(0)
    count_pos = valuelist.count(1)  
    if count_neg>count_pos:
        return 0
    else:
        return 1

def _getmaxvalue(valuelist):
    valuelist = list(valuelist)
    max = 0
    for item in valuelist:
        if item > max:
            max = item
    return max

def Entropy(target_values):
    values = list(target_values)
    elements,counts = _unique(values,True)
    entropy = 0
    sum_counts = _sum(counts)
    for i in elements:
        entropy = entropy + (-counts[i]/sum_counts*math.log2(counts[i]/sum_counts))
    return entropy

def VarianceImpurity(training_set, target_values):
    length_trainingSet = len(training_set)
    pos_count = 0
    neg_count = 0
    for i in range(len(training_set)):
        if target_values[i] == 0:
            neg_count += 1
        elif target_values[i] == 1:
            pos_count += 1
    if pos_count == 0 or neg_count == 0:
        return 0
    variance_impurity = (pos_count/length_trainingSet)*(neg_count/length_trainingSet)
    return variance_impurity

def VarianceImpurityGain(data,split_attr,target_attr='Class'):
    target_values = list(data[target_attr])
    #Calculate total variance impurity of the training set
    total_variance_impurity = VarianceImpurity(data,target_values)
    
    #Calculate variance impurity for the split attribute
    split_attrvalues = list(data[split_attr])
    split_attr_variance_impurity = VarianceImpurity(split_attrvalues, target_values)
    variance_impurity_gain = total_variance_impurity - split_attr_variance_impurity
    return variance_impurity_gain

def InfoGain(data,split_attr,target_attr='Class'):
    #Calculate total entropy of the dataset
    total_entropy = Entropy(data[target_attr])
    
    #Calculate entropy for the split attributes
    values = list(data[split_attr])
    elements,counts = _unique(values,True)
    weighted_entropy = 0
    sum_counts = _sum(counts)
    for i in elements:
        target_values = data[data[split_attr]==i].dropna()[target_attr]
        weighted_entropy = weighted_entropy + (counts[i]/sum_counts)*Entropy(target_values)
    #Calculate Information Gain
    info_gain = total_entropy - weighted_entropy
    #c = {split_attr:info_gain}
    return info_gain

def BuildTree(gainDict):
    root_node = None
    max_infogain = _getmaxvalue(list(gainDict.values()))
    best_attr = [key  for (key, value) in gainDict.items() if value == max_infogain][0]
        #print("best attr : " , best_attr)
    root_node = best_attr
    tree = {root_node:{}}
    return tree,best_attr

def BuildTreeUsingInfoGain(data,attributes,target_attribute='Class'):
        #print(data)
        #if all values are same for target attribute
        target_attr_values = list(data[target_attribute])
        values, counts = _unique(target_attr_values,True)
        if len(values)<=1:
            return values[0]
        #if attributes is empty
        elif len(attributes)==0:
            return _argmaxindex(target_attr_values)
        else:
            infoGainDict = {}
        
            for attribute in attributes:
                info_gain = InfoGain(data,attribute)
                infoGainDict[attribute] = info_gain
        
            #info_gain_values = InfoGain(data,attribute) for attribute in attributes
            #infoGaindictValues = infoGainDict.values()
            tree, best_attr = BuildTree(infoGainDict)
            attributes = [i for i in attributes if i!= best_attr]
            #print("new attributes : " , attributes)
            for value in _unique(list(data[best_attr])):
                sub_data = data[data[best_attr]==value].dropna()
                subtree = BuildTreeUsingInfoGain(sub_data,attributes)
                tree[best_attr][value] = subtree
        return tree
    
def BuildTreeUsingVarImpurityGain(data,attributes,target_attribute='Class'):
    if not(data.isnull().values.any()):
        #if all values are same for target attribute
        target_attr_values = list(data[target_attribute])
        values, counts = _unique(target_attr_values,True)
        if len(values)<=1:
            return values[0]
        #if attributes is empty
        elif len(attributes)==0:
            return _argmaxindex(target_attr_values)
        else:
            
            varianceImpurityGainDict = {}
            for attribute in attributes:
                
                variance_impurity_gain = VarianceImpurityGain(data,attribute)
                varianceImpurityGainDict[attribute] = variance_impurity_gain
                
            tree,best_attr = BuildTree(varianceImpurityGainDict)
            attributes = [i for i in attributes if i!= best_attr]
                #print("new attributes : " , attributes)
            for value in _unique(list(data[best_attr])):
                sub_data = data[data[best_attr]==value].dropna()
                subtree = BuildTreeUsingVarImpurityGain(sub_data,attributes)
                tree[best_attr][value] = subtree
    return tree

def PredictTestData(query,tree,default=1):
    #print(query.keys())
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #print("key is present : ", key)
            #print(tree[key][query[key]])
            try:
                result = tree[key][query[key]] 
                
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return PredictTestData(query,result)
            else:
                return result

def test(testdata,tree):
   #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = testdata.drop('Class', axis=1).to_dict(orient = "records")
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    correct_prediction = 0
    #Calculate the prediction accuracy
    for i in range(len(testdata)):
        predicted.loc[i,"predicted"] = PredictTestData(queries[i],tree,1.0) 
        #print(predicted.loc[i,"predicted"])
        if (predicted.loc[i,"predicted"] == testdata.loc[i,"Class"]):
            correct_prediction += 1
    #print('The prediction accuracy is: ',(_sum(predicted["predicted"] == testdata["Class"])/len(data))*100,'%')
    print("Number of Correct predictions : ", correct_prediction)
    total_prediction = len(testdata)
    accuracy_rate = (correct_prediction/total_prediction)*100
    print("Accuracy of test data is : ", accuracy_rate,"%")
      
tree1 = BuildTreeUsingInfoGain(data, attributes)
#print(tree1)
tree2 = BuildTreeUsingVarImpurityGain(data,attributes) 
#print(tree2)
test(testdata,tree1)
test(testdata,tree2)
