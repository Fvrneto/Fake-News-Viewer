# -*- coding: utf-8 -*-
############################ Importing necessaire modules

import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import ne_chunk
import re
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns

############################ Importing user modules

import Modules_User.cleaning as cleaning 
import Modules_User.frequences as frequences

nltk.download('all')

############################ path to the training data set

path_user = 'C:/Users/Francisco/Desktop/GitHub/'

path_train_fixed = 'Fake-News-Viewer/Data_set/fake-news/train.csv'
path_test_fixed = 'Fake-News-Viewer/Data_set/fake-news/test.csv'
path_test_answer = 'Fake-News-Viewer/Data_set/fake-news/submit.csv'

path_train = path_user + path_train_fixed
path_test = path_user + path_test_fixed
path_answer = path_user + path_test_answer


############################ openin data

df = pd.read_csv(path_train)
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

df
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
print ('\n10 most commun words in the Reliable texts - before cleaning: \n %s' % uncleaned_fdist_text_reliable.most_common(10))      


uncleaned_fdist_title_unreliable = frequences.freq_dist(uncleaned_df_token_unreliable, 'title')
uncleaned_fdist_text_unreliable = frequences.freq_dist(uncleaned_df_token_unreliable, 'text') 

#print ('10 most commun words in the Unreliable title - before cleaning: \n %s' % uncleaned_fdist_title_unreliable.most_common(10))
print ('\n10 most commun words in the Unreliable texts - before cleaning: \n %s' % uncleaned_fdist_text_unreliable.most_common(10))

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
print ('\n10 most commun words in the Reliable texts: \n %s' % fdist_text_reliable.most_common(10))      


fdist_title_unreliable = frequences.freq_dist(df_token_unreliable, 'title')
fdist_text_unreliable = frequences.freq_dist(df_token_unreliable, 'text') 

#print ('10 most commun words in the Unreliable title: \n %s' % fdist_title_unreliable.most_common(10))
print ('\n10 most commun words in the Unreliable texts: \n %s' % fdist_text_unreliable.most_common(10))

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
print ('\n10 most commun POS in the Reliable texts: \n %s' % tags_text_reliable.most_common(10))

tags_title_unreliable = frequences.freq_dist_tags(df_token_unreliable_pos, 'title')
tags_text_unreliable = frequences.freq_dist_tags(df_token_unreliable_pos, 'text')

#print ('10 most commun POS in the Unreliable titles: \n %s' % tags_title_unreliable.most_common(10))
print ('\n10 most commun POS in the Unreliable texts: \n %s' % tags_text_unreliable.most_common(10))


############################ POS - Parts of Speech - plots

#fdist_title_reliable_plot = frequences.dist_freq_plot(tags_title_reliable.most_common(10) , 10, 
                                          # '10 most commun POS in the Reliable titles', 'cyan'')
fdist_text_reliable_plot = frequences.dist_freq_plot(tags_text_reliable , 10, 
                                          '10 most commun POS in the Reliable texts', 'blue')

#fdist_title_unreliable_plot = frequences.dist_freq_plot(tags_title_unreliable.most_common(10) , 10, 
                                           #  '10 most commun POS in the Unreliable titles', 'green')
fdist_text_unreliable_plot = frequences.dist_freq_plot(tags_text_reliable , 10, 
                                          '10 most commun POS in the Urneliable texts', 'red')
plt.show()
plt.close()
   
############################ NE - Named Entitiy

df_token_reliable_ne = df_token_reliable_pos
df_token_reliable_ne['title'] = df_token_reliable_ne['title'].apply(ne_chunk)
df_token_reliable_ne['text'] = df_token_reliable_ne['text'].apply(ne_chunk)


df_token_unreliable_ne = df_token_unreliable_pos
df_token_unreliable_ne['title'] = df_token_unreliable_ne['title'].apply(ne_chunk)
df_token_unreliable_ne['text'] = df_token_unreliable_ne['text'].apply(ne_chunk)


############################ user text

path_user_data = path_user + 'Fake-News-Viewer/User_text/True_user.txt'

df_user_path = open(path_user_data, encoding="utf8")
df_user = df_user_path.read()
df_user_path.close()

############################ cleaning process for user data

df_user_token = word_tokenize(df_user)

stopwords = cleaning.stopwords()

df_user_token = [word for word in df_user_token if not word in stopwords]
df_user_clean = (" ").join(df_user_token)
df_user_clean = cleaning.clean_punctuations(df_user_clean)
df_user_clean = cleaning.clean_numbers(df_user_clean)

############################ changing 0 and 1 to reliable and unreliable

df = pd.read_csv(path_train)
df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

df['label'] = df['label'].map({1:'unreliable', 0: 'reliable'})

############################ leters in text

df['Leters in text'] = df['text'].apply(lambda x: len(x))
sns.set_theme(style="darkgrid")
leters_in_text = sns.boxplot(y='Leters in text', x='label', data=df, palette="Set3")
plt.show()
plt.close()

############################ leters in text, removing outliners

sns.set_theme(style="darkgrid")
leters_in_text = sns.boxplot(y='Leters in text', x='label', data=df, palette="Set3", 
                             showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ words in text

def count_words(text):
    words = len(text.split())
    return words
    
df['Words in text'] = df['text'].apply(lambda x: count_words(x))
sns.set_theme(style="darkgrid")
words_in_text = sns.boxplot(y='Words in text', x='label', data=df, palette="Set3")
plt.show()
plt.close()

############################ words in text, removing outliners
    
sns.set_theme(style="darkgrid")
words_in_text = sns.boxplot(y='Words in text', x='label', data=df, palette="Set3", 
                             showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

########## numbers in text,

def count_numbers(text):
    numbers = len(re.findall('([0-9])',text))
    return numbers

df['Numbers in text'] = df['text'].apply(lambda x: count_numbers(x))
sns.set_theme(style="darkgrid")
numbers_in_text = sns.boxplot(y='Numbers in text', x='label', data=df, palette="Set3")
plt.show()
plt.close()

############################ numbers in text, removing outliners

sns.set_theme(style="darkgrid")
numbers_in_text = sns.boxplot(y='Numbers in text', x='label', data=df, palette="Set3", 
                             showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ anzahl coordinating conjunctions
def count_coor_conj(text):
    cc1 = text.count('and')
    cc2 = text.count('or')
    cc3 = text.count('but')
    cc4 = text.count('for')
    cc5 = text.count('nor')
    cc6 = text.count('yet')
    cc7 = text.count('so')
    coor_conj = cc1+cc2+cc3+cc4+cc5+cc6+cc7
    return coor_conj

df['Coordinating conjunctions'] = df['text'].apply(lambda x: count_coor_conj(x))
sns.set_theme(style="darkgrid")
coord_conjunc = sns.boxplot(y='Coordinating conjunctions', x='label', data=df, palette="Set3")
plt.show()
plt.close()

############################ anzahl coordinating conjunctions, removing outliners

sns.set_theme(style="darkgrid")
coord_conjunc = sns.boxplot(y='Coordinating conjunctions', x='label', data=df, palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ anzahl subordinating conjunctions
def count_sub_conj(text):
    sc1 = text.count('although')
    sc2 = text.count('as')
    sc3 = text.count('as long as')
    sc4 = text.count('because')
    sc5 = text.count('before')
    sc6 = text.count('even if')
    sc7 = text.count('if')
    sc8 = text.count('in order to')
    sc9 = text.count('in case')
    sc10 = text.count('once')
    sc11 = text.count('that')
    sc12 = text.count('though')
    sc13 = text.count('until')
    sc14 = text.count('when')
    sc15 = text.count('whenever')
    sc16 = text.count('wherever')
    sc17 = text.count('while')
    sub_conj = sc1+sc2+sc3+sc4+sc5+sc6+sc7+sc8+sc9+sc10+sc11+sc12+sc13+sc14+sc15+sc16+sc17
    return sub_conj

df['Subordinating conjunctions'] = df['text'].apply(lambda x: count_sub_conj(x))
sns.set_theme(style="darkgrid")
subord_conjunc = sns.boxplot(y='Subordinating conjunctions', x='label', data=df, palette="Set3")
plt.show()
plt.close()

############################ anzahl subordinating conjunctions, removing outliners

sns.set_theme(style="darkgrid")
subord_conjunc = sns.boxplot(y='Subordinating conjunctions', x='label', data=df, palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ leter in text with user text

lt_user = len(df_user)
df_lt_user = pd.DataFrame({'Leters in text':[lt_user],'label':['User text']})
df_lt = df.iloc[:,[5,4]]
df_lt_f = f = pd.concat([df_lt, df_lt_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
leters_in_text = sns.boxplot(y='Leters in text', x='label', data=df_lt_f, palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ words in text

wrd_user = count_words(df_user)
df_wrd_user = pd.DataFrame({'Words in text':[wrd_user],'label':['User text']})
df_wrd = df.iloc[:,[6,4]]
df_wrd_f = pd.concat([df_wrd, df_wrd_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
words_in_text = sns.boxplot(y='Words in text', x='label', data=df_wrd_f, palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ numbers in text

nmb_user = count_numbers(df_user)
df_nmb_user = pd.DataFrame({'Numbers in text':[nmb_user],'label':['User text']})
df_nmb = df.iloc[:,[7,4]]
df_nmb_f = pd.concat([df_nmb, df_nmb_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
words_in_text = sns.boxplot(y='Numbers in text', x='label', data=df_nmb_f, palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ anzahl coordinating conjunctions

cc_user = count_coor_conj(df_user)
df_cc_user = pd.DataFrame({'Coordinating conjunctions':[cc_user],'label':['User text']})
df_cc = df.iloc[:,[8,4]]
df_cc_f = pd.concat([df_cc, df_cc_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
coord_conjunc = sns.boxplot(y='Coordinating conjunctions', x='label', data=df_cc_f , palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()

############################ anzahl subordinating conjunctions

sc_user = count_sub_conj(df_user)
df_sc_user = pd.DataFrame({'Subordinating conjunctions':[sc_user],'label':['User text']})
df_sc = df.iloc[:,[9,4]]
df_sc_f = pd.concat([df_sc, df_sc_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
subord_conjunc = sns.boxplot(y='Subordinating conjunctions', x='label', data=df_sc_f, palette="Set3",
                            showfliers = False, flierprops = dict(markerfacecolor = '0.50', markersize = 2))
plt.show()
plt.close()
