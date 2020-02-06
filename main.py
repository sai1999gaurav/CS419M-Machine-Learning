import pandas as pd
import numpy as np
import csv
import argparse
import sys
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

parser = argparse.ArgumentParser()
parser.add_argument('--train_data',nargs=1)
parser.add_argument('--test_data',	nargs=1)
parser.add_argument('--min_leaf_size',type=int,nargs=1)
args = parser.parse_args()
#print(args.train_data[0])
dataset = pd.read_csv(args.train_data[0])
testing_data = pd.read_csv(args.test_data[0])
min_instance = args.min_leaf_size[0]
#print(args.min_leaf_size[0])
#print(testing_data.shape)


mean_data=np.mean(dataset.iloc[:,-1])

with open('output.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(("Id","output"))


def variance(data,split_attribute_name, target_name = "output"):
    feature_values = np.unique(data[split_attribute_name])
    feature_variance = 0
    for value in feature_values:
        subset = data.query('{0}=={1}'.format(split_attribute_name,value)).reset_index()
        variance_value = len(subset)/len(data)*np.var(subset[target_name], ddof = 1)
        feature_variance += variance_value
    return feature_variance
	
def Classifier(data, originaldata, features,min_instance,target_attribute_name,parent_node_class= None):
    if len(data) <= int(min_instance):
        return np.mean(data[target_attribute_name])
    elif len(data) == 0:
        return np.mean(originaldata[target_attribute_name])
    elif len(features) == 0:
	    return parent_node_class         
    else :
	    parent_node_class = np.mean(data[target_attribute_name])
	    item_values = [variance(data,feature) for feature in features]
	    best_feature_index = np.argmin(item_values)
	    best_feature = features[best_feature_index]
	    tree={best_feature : {}}
	    features=[ i for i in features if i != best_feature]
	    for value in np.unique(data[best_feature]):
	        value=value
	        sub_data = data.where(data[best_feature] == value).dropna()
	        subtree=Classifier(sub_data,originaldata,features,min_instance,"output",parent_node_class = parent_node_class)
	        tree[best_feature][value] = subtree
	    return tree
	    	        
def predict(query,tree,default = mean_data):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
                 
   
training_data = dataset
  


def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = []
    for i in range (len(data)): 
        predicted.append(predict(queries[i],tree,mean_data)) 
    rows = zip(predicted)
    with open('output.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        i=1
        for row in rows:
             writer.writerow([i , *row])
             i=i+1
    csvFile.close()
   
    
    #pd.read_csv('Output.csv').T.to_csv('Output.csv',header=True)
    
    return 0
    
              	    	        

tree = Classifier(training_data, training_data,training_data.columns[:-1],min_instance,"output")
test(testing_data,tree)	




































    	        