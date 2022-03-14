# -*- coding: utf-8 -*-
############################ Importing necessaire modules

import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import ne_chunk

############################ Importing user modules

import Modules_User.cleaning as cleaning 
import Modules_User.frequences as frequences

nltk.download('all')

############################ path to the training data set

path_train = 'C:/Users/franc/Desktop/TechLabs/fake-news/train.csv'

############################ openin data

df = pd.read_csv(path_train)
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ dropping nan:
    
df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

print(df.isna().sum())

############################ tokenization - before cleaning
    
uncleaned_df_token = df

uncleaned_df_token['title'] = uncleaned_df_token['title'].apply(word_tokenize)
uncleaned_df_token['author'] = uncleaned_df_token['author'].apply(word_tokenize)
uncleaned_df_token['text'] = uncleaned_df_token['text'].apply(word_tokenize)
  

############################ spliting the data into reliable and unreliable - before cleaning

uncleaned_df_token_reliable = uncleaned_df_token.drop(df[df.label == 0].index)
uncleaned_df_token_unreliable = uncleaned_df_token.drop(df[df.label == 1].index)
   
############################ word frequencies in titles - before cleaning - before cleaning


uncleaned_fdist_title_reliable = frequences.freq_dist(uncleaned_df_token_reliable, 'title')
uncleaned_fdist_text_reliable = frequences.freq_dist(uncleaned_df_token_reliable , 'text')

#print ('10 most commun words in the Reliable title - before cleaning: \n %s' % uncleaned_fdist_title_reliable.most_common(10)) 
print ('10 most commun words in the Reliable texts - before cleaning: \n %s' % uncleaned_fdist_text_reliable.most_common(10))      


uncleaned_fdist_title_unreliable = frequences.freq_dist(uncleaned_df_token_unreliable, 'title')
uncleaned_fdist_text_unreliable = frequences.freq_dist(uncleaned_df_token_unreliable, 'text') 

#print ('10 most commun words in the Unreliable title - before cleaning: \n %s' % uncleaned_fdist_title_unreliable.most_common(10))
print ('10 most commun words in the Unreliable texts - before cleaning: \n %s' % uncleaned_fdist_text_unreliable.most_common(10))

############################ dist.frequencies plot - before cleaning

#uncleaned_fdist_title_reliable_plot = frequences.dist_freq_plot(fdist_title_reliable , 10, 
#                                           '10 most commun words in the Reliable title - before cleaning', 'cyan')
uncleaned_fdist_text_reliable_plot = frequences.dist_freq_plot(uncleaned_fdist_text_reliable , 10, 
                                          '10 most commun words in the Reliable texts - before cleaning', 'blue')

#uncleaned_fdist_title_unreliable_plot = frequences.dist_freq_plot(fdist_title_unreliable , 10, 
#                                             '10 most commun words in the Unreliable title - before cleaning', 'green')
uncleaned_fdist_text_unreliable_plot = frequences.dist_freq_plot(uncleaned_fdist_text_unreliable , 10, 
                                            '10 most commun words in the Unreliable texts - before cleaning', 'red')

############################ removing stopwords, numbers and punctuations

df_cleaned = pd.read_csv(path_train)
df_cleaned = df_cleaned[df_cleaned['text'].notna()]
df_cleaned.reset_index(drop=True, inplace=True)  
df_cleaned['title'] = df_cleaned['title'].fillna('None')
df_cleaned['author'] = df_cleaned['author'].fillna('None')
df_cleaned = df_cleaned[df_cleaned['text'].notna()]
df_cleaned.reset_index(drop=True, inplace=True)
    
df_cleaned['title'] = df_cleaned['title'].fillna('None')
df_cleaned['author'] = df_cleaned['author'].fillna('None')
df_cleaned = df_cleaned[df_cleaned['text'].notna()]
df_cleaned.reset_index(drop=True, inplace=True)


df_cleaned['title'] = df_cleaned['title'].apply(lambda x: cleaning.clean_numbers(x))
df_cleaned['title'] = df_cleaned['title'].apply(cleaning.clean_steapwords())
df_cleaned['title'] = df_cleaned['title'].apply(lambda x: cleaning.clean_punctuations(x))


df_cleaned['text'] = df_cleaned['text'].apply(lambda x: cleaning.clean_numbers(x))
df_cleaned['text'] = df_cleaned['text'].apply(cleaning.clean_steapwords())
df_cleaned['text'] = df_cleaned['text'].apply(lambda x: cleaning.clean_punctuations(x))


############################ tokenization:
    
df_token = df_cleaned

df_token['title'] = df_token['title'].apply(word_tokenize)
df_token['author'] = df_token['author'].apply(word_tokenize)
df_token['text'] = df_token['text'].apply(word_tokenize)
  

############################ spliting the data into reliable and unreliable

df_token_reliable = df_token.drop(df[df.label == 0].index)
df_token_unreliable = df_token.drop(df[df.label == 1].index)
   
############################ word frequencies


fdist_title_reliable = frequences.freq_dist(df_token_reliable, 'title')
fdist_text_reliable = frequences.freq_dist(df_token_reliable, 'text')

#print ('10 most commun words in the Reliable title: \n %s' % fdist_title_reliable.most_common(10)) 
print ('10 most commun words in the Reliable texts: \n %s' % fdist_text_reliable.most_common(10))      


fdist_title_unreliable = frequences.freq_dist(df_token_unreliable, 'title')
fdist_text_unreliable = frequences.freq_dist(df_token_unreliable, 'text') 

#print ('10 most commun words in the Unreliable title: \n %s' % fdist_title_unreliable.most_common(10))
print ('10 most commun words in the Unreliable texts: \n %s' % fdist_text_unreliable.most_common(10))

############################ dist.frequencies plot



#fdist_title_reliable_plot = frequences.dist_freq_plot(fdist_title_reliable , 10, 
#                                           '10 most commun words in the Reliable title', 'cyan')
fdist_text_reliable_plot = frequences.dist_freq_plot(fdist_text_reliable , 10, 
                                          '10 most commun words in the Reliable texts', 'blue')

#fdist_title_unreliable_plot = frequences.dist_freq_plot(fdist_title_unreliable , 10, 
#                                             '10 most commun words in the Unreliable title', 'green')
fdist_text_unreliable_plot = frequences.dist_freq_plot(fdist_text_unreliable , 10, 
                                            '10 most commun words in the Unreliable texts', 'red')

############################ Stemming

stemmer = SnowballStemmer("english")

df_token_reliable_stemmed = df_token_reliable
df_token_reliable_stemmed['title'] = df_token_reliable['title'].apply(lambda x: [stemmer.stem(y) for 
                                                                                 y in x])  # Stem every word.
df_token_reliable_stemmed['text'] = df_token_reliable['text'].apply(lambda x: [stemmer.stem(y) for 
                                                                               y in x])  # Stem every word.

df_token_unreliable_stemmed = df_token_unreliable
df_token_unreliable_stemmed['title'] = df_token_unreliable['title'].apply(lambda x: [stemmer.stem(y) for 
                                                                                     y in x])  # Stem every word.
df_token_unreliable_stemmed['text'] = df_token_unreliable['text'].apply(lambda x: [stemmer.stem(y) for 
                                                                                   y in x])  # Stem every word.

############################ POS - Parts of Speech

df_token_reliable_pos = df_token_reliable
df_token_reliable_pos['title'] = df_token_reliable_pos['title'].apply(nltk.pos_tag)
df_token_reliable_pos['text'] = df_token_reliable_pos['text'].apply(nltk.pos_tag)


df_token_unreliable_pos = df_token_unreliable
df_token_unreliable_pos['title'] = df_token_unreliable_pos['title'].apply(nltk.pos_tag)
df_token_unreliable_pos['text'] = df_token_unreliable_pos['text'].apply(nltk.pos_tag)

############################ POS - Parts of Speech - frequences

tags_title_reliable = frequences.freq_dist_tags(df_token_reliable_pos, 'title')
tags_text_reliable = frequences.freq_dist_tags(df_token_reliable_pos, 'text')

#print ('10 most commun POS in the Reliable titles: \n %s' % tags_title_reliable.most_common(10))
print ('10 most commun POS in the Reliable texts: \n %s' % tags_text_reliable.most_common(10))

tags_title_unreliable = frequences.freq_dist_tags(df_token_unreliable_pos, 'title')
tags_text_unreliable = frequences.freq_dist_tags(df_token_unreliable_pos, 'text')

#print ('10 most commun POS in the Unreliable titles: \n %s' % tags_title_unreliable.most_common(10))
print ('10 most commun POS in the Unreliable texts: \n %s' % tags_text_unreliable.most_common(10))


############################ POS - Parts of Speech - plots

#fdist_title_reliable_plot = frequences.dist_freq_plot(tags_title_reliable.most_common(10) , 10, 
                                          # '10 most commun POS in the Reliable titles', 'cyan'')
fdist_text_reliable_plot = frequences.dist_freq_plot(tags_text_reliable , 10, 
                                          '10 most commun POS in the Reliable texts', 'blue')

#fdist_title_unreliable_plot = frequences.dist_freq_plot(tags_title_unreliable.most_common(10) , 10, 
                                           #  '10 most commun POS in the Unreliable titles', 'green')
fdist_text_unreliable_plot = frequences.dist_freq_plot(tags_text_unreliable , 10, 
                                            '10 most commun POS in the Unreliable texts', 'red')
   
############################webdavs://webdav.hhu.de/DavWWWRoot/gilch/Daten/Home/Riel Neto/Softwares/Wavelenght_spectra/wavelenght_spectra.pywebdavs://webdav.hhu.de/DavWWWRoot/gilch/Daten/Home/Riel Neto/Softwares/Wavelenght_spectra/wavelenght_spectra.py

df_token_reliable_ne = df_token_reliable_pos
df_token_reliable_ne['title'] = df_token_reliable_ne['title'].apply(ne_chunk)
df_token_reliable_ne['text'] = df_token_reliable_ne['text'].apply(ne_chunk)


df_token_unreliable_ne = df_token_unreliable_pos
df_token_unreliable_ne['title'] = df_token_unreliable_ne['title'].apply(ne_chunk)
df_token_unreliable_ne['text'] = df_token_unreliable_ne['text'].apply(ne_chunk)

