#%%
import pandas as pd 
import os
import numpy as np
pd.set_option('max_colwidth', -1)
wd='file_path'

#%%
os.chdir(wd)
sampled_data = pd.read_csv('sampled_data.csv')
print("shape of imported sampled_data df: %s" % (sampled_data.shape, ))

#%%
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(sampled_data['TriageNotesClean_v2'], sampled_data['Overdose'], test_size=.30)
print("shape of train_x: %s, shape of test_x: %s" % (train_x.shape, test_x.shape))

#%% 
from sklearn.feature_extraction.text import CountVectorizer
# transform the training and validation data using count vectorizer object
def creating_word_vecs(train_x, test_x): 
    count_vect=CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train_x)
    xtrain_count =  count_vect.transform(train_x)
    xtest_count =  count_vect.transform(test_x)
    return xtrain_count, xtest_count

xtrain_count, xtest_count=creating_word_vecs(train_x, test_x)

#%%
#training model with training dataset and predicting overdose on test data  
from sklearn.naive_bayes import MultinomialNB

def train_test_model(xtrain_count, train_y, xtest_count): 
    clf=MultinomialNB().fit(xtrain_count, train_y)
    predictions=clf.predict(xtest_count)
    return predictions

predictions=train_test_model(xtrain_count, train_y, xtest_count)
np.mean(predictions==test_y) 

#%% [markdown]
# Naive Bayes had 95% accuracy which I think is unusually high. i want to test it by comparing visits that were not overdoses from the "possible_overdoses" dataset. I need to clean the triage note text in the same way I cleaned the sampled data dataset.

#%%
possible_overdoses=pd.read_csv('possible_overdoses_tagged.csv')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['pt', 'patient'])
possible_overdoses['TriageNotesClean_v2']=possible_overdoses['TriageNotesClean'].apply(lambda x: " ".join([word for word in str(x).split(" ") if word not in stop_words]))

#%%
# creating dataset with 1 to 1 match od vs not od using suspected od dataset rather than all visits
sample_n = len(possible_overdoses[possible_overdoses['Overdose']==1])
grouped = possible_overdoses.groupby('Overdose')
sampled_data = grouped.apply(lambda x: x.sample(n=sample_n, replace=False))   
train_x, test_x, train_y, test_y = train_test_split(sampled_data['TriageNotesClean_v2'], sampled_data['Overdose'], test_size=.30)
print("shape of train_x: %s, shape of test_x: %s" % (train_x.shape, test_x.shape))

#%%
# transforming training and test text into count vectors 
x_train_count, x_test_count = creating_word_vecs(train_x, test_x)
predictions=train_test_model(xtrain_count, train_y, xtest_count)
np.mean(predictions==test_y)

