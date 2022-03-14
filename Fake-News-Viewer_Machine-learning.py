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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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

best_model = LogisticRegression()
best_model.fit(X_train, y_train)

############################ user text

path_user = 'C:/Users/franc/Desktop/TechLabs/GitHub/Fake-News-Viewer/User_text/True_user.txt'

df_user_path = open(path_user, encoding="utf8")

df_user = df_user_path.read()

df_user_path.close()

############################ cleaning process for user data

df_user_token = word_tokenize(df_user)

stopwords = cleaning.stopwords()

df_user_token = [word for word in df_user_token if not word in stopwords]
df_user = (" ").join(df_user_token)
df_user = cleaning.clean_punctuations(df_user)
df_user = cleaning.clean_numbers(df_user)

############################ predict user text

X_user = cv.transform([df_user])
user_pred = best_model.predict(X_user)
user_pred_value =  user_pred[0]

if user_pred == 0:
    print ("Based on the data set used to fit this algorithm, the informed test is reliable.")
else:
    print ("Based on the data set used to fit this algorithm, the informed test is unreliable.")