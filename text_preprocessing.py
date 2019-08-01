####################################################################################################################################################################################
# this script pulls in data from ESSENCE, does some prelim text cleaning of the triage notes data, spellchecks opioid-related terms of interest and filters a subset of visits that are most likely related to opioid overdose. It saves two non-overlapping datasets to the wd: possible overdose visits and all other visits.   

#enter wd, username, pw, and api url
wd='file_path'
username='username'
pw='pw'
api_link='URL'
file_name='possible_overdoses.csv'

####################################################################################################################################################################################

import pandas as pd 
import requests
import json
import nltk
import fuzzywuzzy
import contractions
from fuzzywuzzy import process
from nltk import word_tokenize
from nltk.util import ngrams
import os

#pulling visits 
urls=api_link

#pulling in data from ESSENCE API
def pulling_dataset(url_list, responses=list()):
    for url in urls:
        response=requests.get(url, auth=(username, pw))
        data = response.json()
        data = data['dataDetails']
        data = pd.DataFrame.from_dict(data, orient='columns')
        responses.append(data)
        return(responses)

responses=pulling_dataset(urls)
ESSENCE_df=responses[0]

#function to conduct prelim cleaning of text data
def text_clean(pandas_series):
    x=pandas_series.str.lower()
    x=x.str.replace(r'[^\w\d\s\']+', ' ', regex=True)
    x=x.apply(lambda y: contractions.fix(y))
    x=x.str.replace('[\']', '', regex=True)
    x=x.str.replace('\s+', ' ')
    x=x.str.strip()
    return(x)

#prelim cleaning text. creates column with semi-clean triage notes text called TriageNotesClean
# punctuation removed, contractions expanded, lower case, extra white spaces removed 
ESSENCE_df['TriageNotesClean']=text_clean(ESSENCE_df['TriageNotesOrig'])

#i want to cast a wide night when tagging cases. in order to do this, I'm going to subset to visits with certain string patterns in CC, DG or triage notes
#before i do this, i want to correct as many typos of key words as possible

# creating dictionary where key is opioid term and value is list of typos
# typos list=used fuzzywuzzy to get terms that have similarity index >0.9 (threshold determined by testing terms in opioid_terms list on this dataset but may vary for other datasets)
# replaced typos with opioid term in TriageNotesClean column

#opioid terms I want to spell check
opioid_terms=['opioid', 'narcan', 'percocet', 'heroin', 'pinpoint', 'oxycontin', 'vicodin', 'overdose', 'oxycodone']

#set of unigrams unigrams
unique_triage_words=set()
ESSENCE_df['TriageNotesClean'].str.split().apply(unique_triage_words.update)

#creating empty dictionary: key=opioid term, values=list of typos
opioid_typos={}

#appending empty dictionary with key value pairs
for word in opioid_terms:
    possible_typos=process.extract(word, unique_triage_words)
    typos=[x[0] for x in possible_typos if x[1]>90 and x[1]!=100]
    if len(typos)>0:
        opioid_typos[word]=typos 

#function replaces typos with correct word using dictionary of typos created above
def replace_typos(text, opioid_typos_dict):
    for key, typos in opioid_typos_dict.items():
        for word in typos: 
            text = text.replace(word, key)
    return text

#applying this to triage notes
ESSENCE_df['TriageNotesClean']=ESSENCE_df['TriageNotesClean'].apply(lambda x:replace_typos(x, opioid_typos))

#creating dataset of possible overdoses
opioid_terms='|'.join(opioid_terms)
possible_overdoses=ESSENCE_df[(ESSENCE_df['TriageNotesClean'].str.contains(opioid_terms)) | (ESSENCE_df['ChiefComplaintOrig'].str.lower().str.contains('overdose|altered mental status|suicide')) | (ESSENCE_df['Diagnosis_Combo'].str.lower().str.contains('poisoning'))]

#creating df of visits that are not possible overdoses
not_possible_overdoses=ESSENCE_df[~ESSENCE_df.C_BioSense_ID.isin(possible_overdoses['C_BioSense_ID'])]
not_possible_overdoses['Overdose']=0
not_possible_overdoses['Unsure']=0

#saving csv file called possible overdoses to wd
os.chdir(wd)
possible_overdoses.to_csv('possible_overdoses.csv')
not_possible_overdoses.to_csv('not_possible_overdoses.csv')