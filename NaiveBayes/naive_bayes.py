
# coding: utf-8

import pandas as pd
import numpy as np
import random
import sys


trainingSetFileName = sys.argv[1]
testSetFileName = sys.argv[2]

trainingData = pd.read_csv(trainingSetFileName)
testData = pd.read_csv(testSetFileName)
testData = testData.fillna("BLANK")


#fill the NaN values with BLANK
trainingData = trainingData.fillna("BLANK")

# determine the number of restaurants that are there
total_restaurants = trainingData['goodForGroups'].count()

#determine the total number of restaurants that are good for groups
total_good_for_groups = trainingData['goodForGroups'][trainingData['goodForGroups'] == 1].count()

total_not_good_for_groups = trainingData['goodForGroups'][trainingData['goodForGroups'] == 0].count()


# determine the probability of a restaurant being good or not good for groups
prob_good_for_groups = total_good_for_groups / total_restaurants
prob_not_good_for_groups = total_not_good_for_groups / total_restaurants


# now we must calculate P(x | y), which is the probability that an attribute is observed
# This is the likelihood
def prob_a_given_b(attribute, value, true):
    total_value = 0
    total_value_count = 0
    prob_a_given_b = 0

    # how many are true/false for the given value 
    # if true is set to 1, then we want for true
    # P(A | B) 
    unique_values = trainingData[attribute].unique()
    unique_values_count = len(unique_values)
    
    if true == 1:
        total_value = trainingData[(trainingData[attribute] == value) & (trainingData.goodForGroups == 1)]
        total_value_count = total_value['goodForGroups'].count()
        prob_a_given_b = (total_value_count + 1) / (total_good_for_groups + unique_values_count)
    else:
        total_value = trainingData[(trainingData[attribute] == value) & (trainingData.goodForGroups == 0)]
        total_value_count = total_value['goodForGroups'].count()
        prob_a_given_b = (total_value_count + 1) / (total_not_good_for_groups + unique_values_count)

    return prob_a_given_b


#determine the columns in our data set
column_lists = list(trainingData.columns.values)

list(trainingData['alcohol'].unique())

# create a dictionary where we can store the probabilities
probability_dictionary = {'first' : 0}

# calculate all the probabilities and store them in a dictionary
# this is the state where we are generating the model
for column in column_lists:
    # determine the unique values
    unique_values = trainingData[column].unique()
    
    # loop through the unique values, calculate their probs, and add them to our structure
    for value in unique_values:
        concatenated_string = str(column) + str(value)

        prob_a_given_b_true = prob_a_given_b(column, value, 1)
        prob_a_given_b_false = prob_a_given_b(column, value, 0)

        
        concatenated_string_true = str(concatenated_string) + str('true')
        concatenated_string_false = str(concatenated_string) + str('false')
        
        #add to the dictionary
        probability_dictionary.update({concatenated_string_true : prob_a_given_b_true})
        probability_dictionary.update({concatenated_string_false : prob_a_given_b_false})


# search for the key in the dictionary
def search_dictionary(key, dictionary):
    if key in dictionary:
        return dictionary.get(key)
    else:
        return ''


search_dictionary('citySaint-Laurenttrue', probability_dictionary)


def make_prediction(current_row):
    squared_loss_function = 0
    column_list = testData.columns
    
    totalTrueProbability = prob_good_for_groups
    totalFalseProbability = prob_not_good_for_groups
    
    goodForGroupsValueHere = 0
    # look at all the values
    for columns in column_list:
        
        row_value = currentRow[columns]
        
        # don't want to use goodForGroups as our predicting value
        if (columns == 'goodForGroups'):
            goodForGroupsHere = int(row_value)
            continue
            
        concatenated_string = str(columns) + str(row_value)
        #display(concatenated_string)
        concatenated_string_true = str(concatenated_string) + str('true')
        concatenated_string_false = str(concatenated_string) + str('false')
        
        # look up the true and false probabilities in the dictionary
        value_true_probability = search_dictionary(concatenated_string_true, probability_dictionary)
        value_false_probability = search_dictionary(concatenated_string_false, probability_dictionary)
        
        # if it doesn't occur, calclulate the probability differently
        if value_true_probability == '':
            unique_values = trainingData[columns].unique() 
            unique_values_count = len(unique_values)
            value_true_probability = 1 / (prob_good_for_groups + unique_values_count)
        if value_false_probability == '':
            unique_values = trainingData[columns].unique() 
            unique_values_count = len(unique_values)
            value_false_probability = 1 / (prob_not_good_for_groups + unique_values_count)
        
        totalTrueProbability = totalTrueProbability * value_true_probability
        totalFalseProbability = totalFalseProbability * value_false_probability
        
    
    
    if (totalTrueProbability > totalFalseProbability):
        goodForGroupsValue = 1
    else:
        goodForGroupsValue = 0

    if goodForGroupsHere == 1:
        currentProbability = (totalTrueProbability) / (totalTrueProbability + totalFalseProbability)
        squared_loss_function = (1 - currentProbability)
        squared_loss_function = squared_loss_function ** 2
    else:
        currentProbability = (totalFalseProbability) / (totalTrueProbability + totalFalseProbability)
        squared_loss_function = 1 - currentProbability
        squared_loss_function = squared_loss_function ** 2

    return goodForGroupsValue, squared_loss_function


zero_one_loss = 0
squared_loss = 0
for index, currentRow, in testData.iterrows():
    result, squared_loss_value = make_prediction(currentRow)
    goodForGroupsValue = currentRow[0]
    if result == goodForGroupsValue:
        zero_one_loss = zero_one_loss + 0
    else:
        zero_one_loss = zero_one_loss + 1

    squared_loss = squared_loss + squared_loss_value

total_restaurants_test = testData['goodForGroups'].count()
zero_one_loss = zero_one_loss / total_restaurants_test
print("ZERO_ONE LOSS:")
print(zero_one_loss)

squared_loss = squared_loss / total_restaurants_test
print("SQUARED LOSS:")
print(squared_loss)



