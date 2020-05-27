"""
Created on Mon May 25 17:52:35 2020

@author: Priyanshi
"""

import csv
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy

with open('ra_data_classifier.csv', newline='', errors='ignore') as f:
    reader = csv.reader(f)
    train_file = list(reader)
df = DataFrame(train_file,columns=['hid','chunk','has_space'])
del df['hid']
df = df.iloc[1:]
for chunks in df['chunk']:
    strs=''
    for words in range(0,len(chunks)):
        document = re.sub("[^a-zA-Z ]+", '', str(chunks[words]))
        strs=strs+document
    df.replace(chunks,strs,inplace=True)
df_test = df.tail(10)
df_test_result = df_test[:]
del df_test['has_space']
del df_test_result['chunk']
df = df[:90]
count_0 = df.has_space.value_counts()['0']
count_1 = df.has_space.value_counts()['1']
prob_count_0 = count_0/(count_0+count_1)
prob_count_1 = count_1/(count_0+count_1)
docs_0 = [row['chunk'] for index,row in df.iterrows() if row['has_space'] == '0']
vec_0 = CountVectorizer()
X_0 = vec_0.fit_transform(docs_0)
tdm_0 = DataFrame(X_0.toarray(), columns=vec_0.get_feature_names())

docs_1 = [row['chunk'] for index,row in df.iterrows() if row['has_space'] == '1']
vec_1 = CountVectorizer()
X_1 = vec_1.fit_transform(docs_1)
tdm_1 = DataFrame(X_1.toarray(), columns=vec_1.get_feature_names())

word_list_0 = vec_0.get_feature_names();    
count_list_0 = X_0.toarray().sum(axis=0) 
freq_0 = dict(zip(word_list_0,count_list_0))

word_list_1 = vec_1.get_feature_names();    
count_list_1 = X_1.toarray().sum(axis=0) 
freq_1 = dict(zip(word_list_1,count_list_1))

prob_0 = []
for count in count_list_0:
    prob_0.append(count/len(word_list_0))

prob_1 = []
for count in count_list_1:
    prob_1.append(count/len(word_list_1))

docs = [row['chunk'] for index,row in df.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())

total_cnts_features_0 = count_list_0.sum(axis=0)
total_cnts_features_1 = count_list_1.sum(axis=0)

final_result = []
for chunks in df_test['chunk']:
    prob_0_test = []
    prob_1_test = []
    splits = chunks.split()
    for words in splits:
        if words in freq_0.keys():
            count = freq_0[words]
        else:
            count = 0.0
        if words in freq_1.keys():
            count1 = freq_1[words]
        else:
            count1 = 0.0
        prob_0_test.append((count+1.0)/(total_cnts_features_0 + total_features))
        prob_1_test.append((count1+1.0)/(total_cnts_features_1 + total_features))
    multFact = numpy.prod(prob_0_test)
    multFact1 = numpy.prod(prob_1_test)
    check_for_0 = multFact * prob_count_0
    check_for_1 = multFact1 * prob_count_1
    if check_for_0 > check_for_1:
        final_result.append('0')
    else:
        final_result.append('1')

df_test_actual = df_test_result['has_space'].values.tolist()

counter = 0
for index in range(0,len(final_result)):
    if final_result[index] == df_test_actual[index]:
        counter=counter+1
counter = 1.0 * (counter*100/(len(final_result)))
print("Accuracy is ",counter)