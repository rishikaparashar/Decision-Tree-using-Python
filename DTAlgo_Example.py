# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:57:25 2018

@author: Rishika
"""
import pandas as pd
import math
import sys
import pprint
from random import randint
import copy

L = sys.argv[1]
K = sys.argv[2]
training_set_path = sys.argv[3]
validation_set_path = sys.argv[4]
test_set_path = sys.argv[5]
to_print = sys.argv[6]
data = pd.read_csv(training_set_path)
testdata = pd.read_csv(test_set_path)
validationData = pd.read_csv(validation_set_path)
attributes = list(data.drop('Class', axis=1))
target_attr_values = list(data['Class'])
positive = 1
negative = 0
n = 0
node_number = 0


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

def MajorityClass():
    best_class = None
    positiveCount = data['Class'].value_counts()[1];
    negativeCount = data['Class'].value_counts()[0];
    if positiveCount>negativeCount :
        best_class = 1
    elif positiveCount<negativeCount:
        best_class = 0
    else:
        best_class = randint(0,1)
    return best_class
    
        
def Entropy(target_values):
    values = list(target_values)
    elements,counts = _unique(values,True)
    entropy = 0
    sum_counts = _sum(counts)
    for i in elements:
        entropy = entropy + (-counts[i]/sum_counts*math.log2(counts[i]/sum_counts))
    return entropy

def VarianceImpurity(target_values):
    values = list(target_values)
    elements,counts = _unique(values,True)
    variance_impurity = 0
    sum_counts = _sum(counts)
    for i in elements:
        variance_impurity = variance_impurity + (-counts[i]/sum_counts*(counts[i]/sum_counts))
    return variance_impurity

def VarianceImpurityGain(data,split_attr,target_attr='Class'):
    target_values = list(data[target_attr])
    #Calculate total variance impurity of the training set
    total_variance_impurity = VarianceImpurity(target_values)
    #Calculate variance impurity for the split attribute
    weighted_variance_impurity = 0
    values = list(data[split_attr])
    elements,counts = _unique(values,True)
    sum_counts = _sum(counts)
    for i in elements:
        split_target_values = data[data[split_attr]==i].dropna()[target_attr]
        weighted_variance_impurity = weighted_variance_impurity + (counts[i]/sum_counts)*VarianceImpurity(split_target_values)
    #Calculate Information Gain
    variance_impurity_gain = total_variance_impurity - weighted_variance_impurity
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
    return info_gain

def BuildTree(gainDict):
    global node_number
    root_node = None
    max_infogain = _getmaxvalue(list(gainDict.values()))
    best_attr = [key  for (key, value) in gainDict.items() if value == max_infogain][0]
    root_node = best_attr
    tree = {root_node:{}}
    positiveCount = data['Class'].value_counts()[1];
    negativeCount = data['Class'].value_counts()[0];
    if positiveCount>negativeCount :
        best_class = 1
    elif positiveCount<negativeCount:
        best_class = 0
    else:
        best_class = randint(0,1)
    #tree[best_attr]['majorClass'] = best_class
    node_number = node_number + 1
    #tree[best_attr]['number'] = node_number

    return tree,best_attr

def BuildTreeUsingInfoGain(data,attributes,target_attribute='Class'):
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
            tree, best_attr = BuildTree(infoGainDict)
            attributes = [i for i in attributes if i!= best_attr]
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
            for attr_val, data_subset in data.groupby(best_attr):
                subtree = BuildTreeUsingVarImpurityGain(data_subset,attributes)
                tree[best_attr][attr_val] = subtree
    return tree

nodeCount = 1
def preorder (temptree, number):
    global nodeCount
    #print(nodeCount)
    if isinstance(temptree, dict):
        attribute = list(temptree.keys())[0]
        if nodeCount == number:
            #print(tree[attribute][0])
            #print(tree[attribute][1])
            if(temptree[attribute][0]!=0 and temptree[attribute][0]!=1):
                #print('here')
                temp_tree = temptree[attribute][0]
                temp_attribute = list(temp_tree.keys())[0]
                temptree[attribute][0] = MajorityClass()
            elif(temptree[attribute][1]!=0 and temptree[attribute][1]!=1):
                #print('here')
                temp_tree = temptree[attribute][1]
                if isinstance(temp_tree, dict):
                    temp_attribute = list(temp_tree.keys())[0]       
                    temptree[attribute][1] = MajorityClass()
        else:
            nodeCount = nodeCount + 1
            left = temptree[attribute][0]
            right = temptree[attribute][1]
            preorder(left, number)
            preorder(right,number )
    return temptree

def NonLeafNodes(tree):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        left = tree[attribute][0]
        right = tree[attribute][1]
        return (1 + NonLeafNodes(left) + NonLeafNodes(right))
    else:
        return 0;
    
def PostPruneTree(L, K, tree):
    bestTree = tree
    #accuracyBeforePruning = test(validationData,best_tree)
    for i in range(1, L+1) :
        D = copy.deepcopy(bestTree)
        M = randint(1, K);
        for j in range(1, M+1):
            count = NonLeafNodes(D)
            if count> 0:
                P = randint(1,count)
            else:
                P = 0
            preorder(D, P)
        accuracyBeforePruning = test(validationData,bestTree) 
        #print("accuracy before pruning on validation data : ",accuracyBeforePruning)
        postPruneAccuracy = test(validationData,D)
        #print("accuracy after pruning on validation data : ",postPruneAccuracy)
        if postPruneAccuracy >= accuracyBeforePruning:
            bestTree = D
    return bestTree

def PredictTestData(query,tree,default=1):  
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return PredictTestData(query,result)
            else:
                return result

def test(testdata,tree):
    queries = testdata.drop('Class', axis=1).to_dict(orient = "records")
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    correct_prediction = 0
    #Calculate the prediction accuracy
    for i in range(len(testdata)):
        predicted.loc[i,"predicted"] = PredictTestData(queries[i],tree,1.0) 
        if (predicted.loc[i,"predicted"] == testdata.loc[i,"Class"]):
            correct_prediction += 1
    total_prediction = len(testdata)
    accuracy_rate = (correct_prediction/total_prediction)*100
    #print("Accuracy of test data is : ", accuracy_rate,"%")
    return accuracy_rate

invalidKeys = {'majorClass','number'}
def printTree(tree):
    for key in list(tree.keys()):
        val = tree[key]
        keysList = []
        if isinstance(val,dict):
            for i in range(0,1):
                keysList = list(val.keys()[i])
                tree = tree[key][keysList[i]]
        printTree(tree)
    print(tree)
    
infoGainTree = BuildTreeUsingInfoGain(data, attributes)
varianceTree = BuildTreeUsingVarImpurityGain(data,attributes) 
accuracy_before_pruning_infogain = test(testdata,infoGainTree)
print("Accuracy before pruning with info gain: ",accuracy_before_pruning_infogain, "%")
accuracy_before_pruning_variance = test(testdata,varianceTree)
print("Accuracy before pruning with variance gain: ",accuracy_before_pruning_variance, "%")
postPruneTreeInfoGain = PostPruneTree(1,9,infoGainTree)
accuracy_after_pruning_infoGain = test(validationData,postPruneTreeInfoGain)
print("Accuracy after pruning with info gain : ",accuracy_after_pruning_infoGain, "%")
postPruneTreeVarianceGain = PostPruneTree(1,9,varianceTree)
accuracy_after_pruning_variance = test(validationData,postPruneTreeVarianceGain)
print("Accuracy after pruning with variance gain : ",accuracy_after_pruning_variance, "%")

#pprint.pprint(infoGainTree)
if to_print == 'yes':
    print("Tree with info gain before pruning :\n")
    pprint.pprint(infoGainTree)
    print("Tree with variance gain before pruning :\n")
    pprint.pprint(varianceTree)
    print("Tree with info gain after pruning :\n")
    pprint.pprint(postPruneTreeInfoGain)
    print("Tree with variance gain after pruning :\n")
    pprint.pprint(postPruneTreeVarianceGain)
    


