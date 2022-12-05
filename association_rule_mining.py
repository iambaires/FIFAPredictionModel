#-------------------------------------------------------------------------
# AUTHOR: William Baires
# FILENAME: association_rule_mining.py
# FOR: CS 5990
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)

encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        labels[item] = 0
        for i in range(len(row)):
            if row[i] == item:
                labels[item] = 1
                break
    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
for index, rule in rules.iterrows():

    print("")
    print(f',{list(rule.antecedents)} + -> , {list(rule.consequents)}')
    print("Support:" + str(rule.support))
    print("Confidence:" + str(rule.confidence))

    support_count = 0
    for idx, row in df.iterrows():
        found = True
        for val in list(rule.antecedents):
            if val not in row.values:
                found = False
        if found is True:
            support_count += 1

    prior = support_count / len(encoded_vals)
    print("Prior:"+str(prior))
    print("Gain in Confidence: " + str(100*(rule.confidence-prior)/prior))
#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()













