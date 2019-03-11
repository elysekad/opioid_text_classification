#%%
import pandas as pd 
import os
pd.set_option('max_colwidth', -1)
wd='C:\\Users\\eak1303\\opioid_text_classification\\everett_sample'

#%%
sampled_data = pd.read_csv('sampled_data.csv')

#%%
from sklearn import model_selection
train_x, test_x, train_y, test_y = model_selection.train_test_split(sampled_data['TriageNotesClean_v2'], sampled_data['Overdose'])

#%% 
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_x)

# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(train_x)
xtest_count =  count_vect.transform(test_x)