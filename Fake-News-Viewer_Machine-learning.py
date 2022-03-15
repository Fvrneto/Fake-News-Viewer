# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:32:32 2022

@author: Francisco, Mimi, Lukas, Ibrahim
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk import word_tokenize
import re
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns


############################ download nltk

import nltk
nltk.download('all')

############################ Importing user modules

import Modules_User.cleaning as cleaning 

############################ 
 
path_user = 'C:/Users/franc/Desktop/TechLabs/GitHub/'

path_train_fixed = 'Fake-News-Viewer/Data_set/fake-news/train.csv'
path_test_fixed = 'Fake-News-Viewer/Data_set/fake-news/test.csv'
path_test_answer = 'Fake-News-Viewer/Data_set/fake-news/submit.csv'

path_train = path_user + path_train_fixed
path_test = path_user + path_test_fixed
path_answer = path_user + path_test_answer


df = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df_answer = pd.read_csv(path_answer)

df_test = pd.merge(df_test, df_answer)

############################ dropping nan:

df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)


df_test['title'] = df_test['title'].fillna('None')
df_test['author'] = df_test['author'].fillna('None')
df_test = df_test[df_test['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ Machine Learning ############################
############################                  ############################

############################ removing stopwords, numbers and punctuations

df_clean = df

df_clean['text'] = df_clean['text'].apply(lambda x: cleaning.clean_numbers(x))
df_clean['text'] = df_clean['text'].apply(cleaning.clean_steapwords())
df_clean['text'] = df_clean['text'].apply(lambda x: cleaning.clean_punctuations(x))


df_test_clean = df_test

df_test_clean['text'] = df_test_clean['text'].apply(lambda x: cleaning.clean_numbers(x))
df_test_clean['text'] = df_test_clean['text'].apply(cleaning.clean_steapwords())
df_test_clean['text'] = df_test_clean['text'].apply(lambda x: cleaning.clean_punctuations(x))

############################ Prepare for Machine Learning: Count Vectorizer

cv = TfidfVectorizer(min_df=1)

df_X = cv.fit(df_clean['text'])
df_X = cv.transform(df_clean['text'])

df_test_X = cv.transform(df_test_clean['text'])

df_y = df["label"].values

X_train = df_X
y_train = df_y

df_test_y = df_test_clean["label"].values

X_test = df_test_X 
y_test = df_test_y

Models = [LogisticRegression(),
          MultinomialNB(),
          RandomForestClassifier(),
          DecisionTreeClassifier(),
          AdaBoostClassifier()]

Models_text = ['LogisticRegression',
                'MultinomialNB',
                'RandomForestClassifier',
                'DecisionTreeClassifier',
                'AdaBoostClassifier']

Model_accuracy = []
Model_test = []
Precision = []
Recall = []
F1_score = []        
           
for model_name in Models:
    
    model_name.fit(X_train, y_train)

    y_pred_test = model_name.predict(X_train)
    y_expect_test = y_train
    accuracy_model = accuracy_score(y_expect_test, y_pred_test)
    Model_accuracy.append(accuracy_model) 
    
    y_pred = model_name.predict(X_test)
    y_expect = y_test
    accuracy_test = accuracy_score(y_expect, y_pred)
    Model_test.append(accuracy_test)
    
    precision = precision_score(y_expect, y_pred)
    Precision.append(precision)

    recall = recall_score(y_expect, y_pred)
    Recall.append(recall)
    
    f1_score_v = f1_score(y_expect, y_pred)
    F1_score.append(f1_score_v)
    
df_models = pd.DataFrame({'Models': Models_text, 
                         'Accuracy': Model_accuracy , 
                         'Test': Model_test,
                         'Precision': Precision,    
                         'Recall': Recall, 
                         'F1_scorel': F1_score
                         })

print(df_models)

############################ Using best model = LogisticRegression() to predict user text

best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

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

############################ predict user text

X_user = cv.transform([df_user_clean])
user_pred = best_model.predict(X_user)
user_pred_value = user_pred[0]

if user_pred == 0:
    print ("\nBased on the data set used to fit this algorithm, the informed test is reliable.")
else:
    print ("\nBased on the data set used to fit this algorithm, the informed test is unreliable.")

############################ Plotting step ############################
############################               ############################

############################ changing 0 and 1 to reliable and unreliable

df = pd.read_csv(path_train)
df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)


df_test['title'] = df_test['title'].fillna('None')
df_test['author'] = df_test['author'].fillna('None')
df_test = df_test[df_test['text'].notna()]
df.reset_index(drop=True, inplace=True)

df['label'] = df['label'].map({1:'unreliable', 0: 'reliable'})

############################ leters in text

df['Leters in text'] = df['text'].apply(lambda x: len(x))
sns.set_theme(style="darkgrid")
leters_in_text = sns.boxplot(y='Leters in text', x='label', data=df, palette="Set3")
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

############################ numbers in text

def count_numbers(text):
    numbers = len(re.findall('([0-9])',text))
    return numbers

df['Numbers in text'] = df['text'].apply(lambda x: count_numbers(x))
sns.set_theme(style="darkgrid")
numbers_in_text = sns.boxplot(y='Numbers in text', x='label', data=df, palette="Set3")
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

############################ leter in text with user text

lt_user = len(df_user)
df_lt_user = pd.DataFrame({'Leters in text':[lt_user],'label':['User text']})
df_lt = df.iloc[:,[5,4]]
df_lt_f = f = pd.concat([df_lt, df_lt_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
leters_in_text = sns.boxplot(y='Leters in text', x='label', data=df_lt_f, palette="Set3")
plt.show()
plt.close()

############################ words in text

wrd_user = count_words(df_user)
df_wrd_user = pd.DataFrame({'Words in text':[wrd_user],'label':['User text']})
df_wrd = df.iloc[:,[6,4]]
df_wrd_f = pd.concat([df_wrd, df_wrd_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
words_in_text = sns.boxplot(y='Words in text', x='label', data=df_wrd_f, palette="Set3")
plt.show()
plt.close()

############################ numbers in text

nmb_user = count_numbers(df_user)
df_nmb_user = pd.DataFrame({'Numbers in text':[nmb_user],'label':['User text']})
df_nmb = df.iloc[:,[7,4]]
df_nmb_f = pd.concat([df_nmb, df_nmb_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
words_in_text = sns.boxplot(y='Numbers in text', x='label', data=df_nmb_f, palette="Set3")
plt.show()
plt.close()

############################ anzahl coordinating conjunctions

cc_user = count_coor_conj(df_user)
df_cc_user = pd.DataFrame({'Coordinating conjunctions':[cc_user],'label':['User text']})
df_cc = df.iloc[:,[8,4]]
df_cc_f = pd.concat([df_cc, df_cc_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
coord_conjunc = sns.boxplot(y='Coordinating conjunctions', x='label', data=df_cc_f , palette="Set3")
plt.show()
plt.close()

############################ anzahl subordinating conjunctions

sc_user = count_sub_conj(df_user)
df_sc_user = pd.DataFrame({'Subordinating conjunctions':[sc_user],'label':['User text']})
df_sc = df.iloc[:,[9,4]]
df_sc_f = pd.concat([df_sc, df_sc_user]).reset_index(drop=True)

sns.set_theme(style="darkgrid")
subord_conjunc = sns.boxplot(y='Subordinating conjunctions', x='label', data=df_sc_f, palette="Set3")
plt.show()
plt.close()

