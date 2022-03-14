# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:32:01 2022

@author: franc
"""
import re
import string
from nltk.corpus import stopwords

stopw = stopwords.words('english')

new_stepw = list()

for word in stopw:
    if "'" in word:
        word_new1 = word.replace("'","’")
        word_new2 = word.replace("'","‘")
        new_stepw.append(word_new1)
        new_stepw.append(word_new2)

stopw.extend(new_stepw)

punct = string.punctuation
punct = punct + "‘" + "’" + '“' + '”' + '—' + "–"

def stopwords():
    
    text=stopw
    
    return(text)

def clean_punctuations(text):
    
    text = re.sub('[%s]' % punct, '', text)

    return(text)

def clean_numbers(text):
    
    text = re.sub(r'[0-9]', '', text)

    return(text)

def clean_steapwords():
    
    clean_lambda = lambda x: ' '.join([word for word in x.split() if word not in (stopw)])
    
    return(clean_lambda)