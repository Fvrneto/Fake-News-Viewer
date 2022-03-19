# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:32:32 2022

@author: Francisco, Mimi, Lukas, Ibrahim
"""

import pandas as pd
import numpy as np

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
from  sklearn.model_selection import RandomizedSearchCV

############################ download nltk

import nltk
nltk.download('all')

############################ Importing user modules

import Modules_User.cleaning as cleaning 

############################ path
 
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
          AdaBoostClassifier()
          ]

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

############################ Testing parameters to reduce overfitting on selected models

n_estimators = np.arange(20,105,5).tolist()
max_features = np.arange(10,75,5).tolist()
min_samples_leaf = np.arange(1,5).tolist()

param_rnd= {
           'max_depth': [6, 9, None], 
           'n_estimators': n_estimators, 
           'max_features': max_features,
           'criterion' : ['gini', 'entropy'],
           'min_samples_leaf': min_samples_leaf
           }
 
rnd_search = RandomizedSearchCV(RandomForestClassifier(), param_rnd, n_iter = 10)
rnd_search.fit(X_train,y_train)
rnd_param = rnd_search.best_params_

max_iter = np.arange(100,900,100).tolist()

param_ld = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'max_iter': [100, 1000, 2500, 5000],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }

ld_search = RandomizedSearchCV(LogisticRegression(), param_ld, n_iter = 10)
ld_search.fit(X_train,y_train)
ld_param = ld_search.best_params_

param_abc = {
            'n_estimators': [100, 150, 200, 250, 500],
            'learning_rate': [0.001, 0.01, 0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
             }

Models_reduce = [LogisticRegression(penalty= ld_param['penalty'],
                                    C = ld_param['C'],
                                    solver= ld_param['solver'], 
                                    max_iter = ld_param['max_iter']),
                 RandomForestClassifier(n_estimators = rnd_param['n_estimators'],
                                        max_depth = rnd_param['max_depth'],
                                        max_features = rnd_param['max_features'],
                                        criterion = rnd_param['criterion'],
                                        min_samples_leaf = rnd_param['min_samples_leaf']
                                        ),
                 ]

Models_text = ['Reduce overfitting LogisticRegression',
               'Reduce overfitting RandomForestClassifier',             
               ]

Model_accuracy = []
Model_test = []
Precision = []
Recall = []
F1_score = []        
           

for model_name in Models_reduce:
    
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
    
df_models_reduce = pd.DataFrame({'Models': Models_text, 
                         'Accuracy': Model_accuracy , 
                         'Test': Model_test,
                         'Precision': Precision,    
                         'Recall': Recall, 
                         'F1_scorel': F1_score
                         })

print(df_models_reduce)


############################ Using best model = LogisticRegression() to predict user text

best_model = RandomForestClassifier(n_estimators = rnd_param['n_estimators'],
                                    max_depth = rnd_param['max_depth'],
                                    max_features = rnd_param['max_features'],
                                    criterion = rnd_param['criterion'],
                                    min_samples_leaf = rnd_param['min_samples_leaf']
                                    )
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
    print ("\nBased on the data set used to fit this algorithm, the informed user text is reliable.")
else:
    print ("\nBased on the data set used to fit this algorithm, the informed user tex is unreliable.")



