# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:32:32 2022

@author: franc
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle

############################ download stopwords

import nltk
nltk.download('stopwords')

############################ Importing user modules

import Modules_User.cleaning as cleaning 

############################ 
path_train = 'C:/Users/franc/Desktop/TechLabs/GitHub/Fake-News-Viewer/fake-news/train.csv'

df = pd.read_csv(path_train)

############################ dropping nan:

df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ removing stopwords, numbers and punctuations

df_clean = df

df_clean['text'] = df_clean['text'].apply(lambda x: cleaning.clean_numbers(x))
df_clean['text'] = df_clean['text'].apply(cleaning.clean_steapwords())
df_clean['text'] = df_clean['text'].apply(lambda x: cleaning.clean_punctuations(x))

############################ Prepare for Machine Learning: Count Vectorizer

cv = TfidfVectorizer(min_df=1)

df_X = cv.fit(df_clean['text'])
df_X = cv.transform(df_clean['text'])

print(df_clean['text'].head())

print('Full vector: ')
print(df_X.toarray)

df_y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

Models = [LogisticRegression(),
          MultinomialNB(),
          RandomForestClassifier(),
          DecisionTreeClassifier(),
          AdaBoostClassifier()]

Model_accuracy = []
Model_test = []
                   
for model_name in Models:
    
    model_name.fit(X_train, y_train)

    y_pred = model_name.predict(X_test)
    y_expect = y_test
    accuracy_model = accuracy_score(y_expect, y_pred)
    Model_accuracy.append(accuracy_model)

    y_pred_test = model_name.predict(X_train)
    y_expect_test = y_train
    accuracy_test = accuracy_score(y_expect_test, y_pred_test)
    Model_test.append(accuracy_test) 
    
df_models = pd.DataFrame({'Models': Models, 
                         'Accuracy': Model_accuracy , 
                         'Test': Model_test, 
                         })

print(df_models)

############################ Using best model = LogisticRegression() to predict user text

best_model = LogisticRegression()
best_model.fit(X_train, y_train)

############################ user text

path_user_fake = 'C:/Users/franc/Desktop/TechLabs/GitHub/Fake-News-Viewer/other_data_set/archive/Fake.csv'
path_user_true = 'C:/Users/franc/Desktop/TechLabs/GitHub/Fake-News-Viewer/other_data_set/archive/True.csv'

df_user_true = pd.read_csv(path_user_true)
df_user_fake = pd.read_csv(path_user_fake)

############################ cleaning process

df_user_true = df_user_true["text"].apply(cleaning.clean_steapwords())
df_user_fake = df_user_fake["text"].apply(cleaning.clean_steapwords())

df_user_true = df_user_true.iloc[0]
df_user_fake = df_user_fake.iloc[1]

df_user_true = cleaning.clean_punctuations(df_user_true)
df_user_true = cleaning.clean_punctuations(df_user_true)


df_user_fake = cleaning.clean_punctuations(df_user_fake)
df_user_fake = cleaning.clean_punctuations(df_user_fake)

############################ predict user text

X_user_true = cv.transform([df_user_true])
user_pred_true = best_model.predict(X_user_true)
print(user_pred_true)

X_user_fake = cv.transform([df_user_fake])
user_pred_fake = best_model.predict(X_user_fake)
print(user_pred_fake)
