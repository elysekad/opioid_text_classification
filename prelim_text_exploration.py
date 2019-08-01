
#%% 
#set up
pd.set_option('max_colwidth', -1)
import pandas as pd 
import os
wd='file_path'

#%% 
# importing possible overdose tagged dataset and not possible overdose dataset
os.chdir(wd)
possible_od_tagged = pd.read_csv('possible_overdoses_tagged.csv')
not_possible_od = pd.read_csv('not_possible_overdoses.csv')
all_visits=pd.concat([possible_od_tagged, not_possible_od])

#%% 
# creating dataset with 1 to 1 match od vs not od
sample_n = len(possible_od_tagged[possible_od_tagged['Overdose']==1])
grouped = all_visits.groupby('Overdose')
sampled_data = grouped.apply(lambda x: x.sample(n=sample_n, replace=False))   

#%% [markdown]
 ### prelim text exploration
 # In this section I want to remove stop words, possibly remove common words and rare words and look at some basic descriptives of the text.  

#%%
#average number of words per triage note by overdose group
grouped['TriageNotesClean'].apply(lambda x: len(str(x).split(" "))/sample_n)

#%%
#removing stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
sampled_data['TriageNotesClean_v2'] = sampled_data['TriageNotesClean'].apply(lambda x: " ".join([word for word in x.split(" ") if word not in stop_words]))

#%%
from collections import Counter
sampled_data['TriageNotesClean'].str.split(" ").value_counts()
word_freq=sampled_data['TriageNotesClean_v2'].str.split(expand=True).stack().value_counts()
word_freq.head(10)

#%% [markdown]
#the only common words that I feel comfortable removing at this point are "pt" and "patient". 

#%%
remove_words = ['pt', 'patient']
sampled_data['TriageNotesClean_v2'] = sampled_data['TriageNotesClean_v2'].apply(lambda x: ' '.join([word for word in x.split(' ') if word not in remove_words]))

#%%
#creating word tokens 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def freq_dist_plot(string):
    word_tokens=word_tokenize(string)
    word_dist=FreqDist(word_tokens)
    word_dist.plot(20)

od_string = sampled_data['TriageNotesClean_v2'][sampled_data['Overdose']==1].str.cat()
not_od_string = sampled_data['TriageNotesClean_v2'][sampled_data['Overdose']==0].str.cat()

freq_dist_plot(od_string)
freq_dist_plot(not_od_string)
