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
from nltk import word_tokenize

############################ download stopwords

import nltk
nltk.download('stopwords')

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

print(df_clean['text'].head())

print('Full vector: ')
print(df_X.toarray)

df_test_X = cv.transform(df_test_clean['text'])

print(df_test_clean['text'].head())

print('Full vector: ')
print(df_X.toarray)

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

Model_accuracy = []
Model_test = []
                   
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

    
df_models = pd.DataFrame({'Models': Models, 
                         'Accuracy': Model_accuracy , 
                         'Test': Model_test, 
                         })

print(df_models)

############################ Using best model = LogisticRegression() to predict user text

best_model = LogisticRegression()
best_model.fit(X_train, y_train)

############################ user text

path_user_true = 'C:/Users/franc/Desktop/TechLabs/GitHub/Fake-News-Viewer/User_text/True_user.txt'
path_user_fake = 'C:/Users/franc/Desktop/TechLabs/GitHub/Fake-News-Viewer/User_text/Fake_user.txt'


df_user_true_path = open(path_user_true, encoding="utf8")
df_user_fake_path = open(path_user_fake, encoding="utf8")


df_user_true = df_user_true_path.read()
df_user_fake = df_user_fake_path.read()

df_user_true_path.close()
df_user_fake_path.close()


############################ cleaning process for user data

df_user_true_token = word_tokenize(df_user_true)
df_user_fake_token = word_tokenize(df_user_fake)

stopwords = cleaning.stopwords()

df_user_true_token = [word for word in df_user_true_token if not word in stopwords]
df_user_true = (" ").join(df_user_true_token)
df_user_true = cleaning.clean_punctuations(df_user_true)
df_user_true = cleaning.clean_numbers(df_user_true)


df_user_fake_token = [word for word in df_user_fake_token if not word in stopwords]
df_user_fake = (" ").join(df_user_fake_token)
df_user_fake = cleaning.clean_punctuations(df_user_fake)
df_user_fake = cleaning.clean_numbers(df_user_fake)

############################ predict user text

X_user_true = cv.transform([df_user_true])
user_pred_true = best_model.predict(X_user_true)
print(user_pred_true)

X_user_fake = cv.transform([df_user_fake])
user_pred_fake = best_model.predict(X_user_fake)
print(user_pred_fake)
