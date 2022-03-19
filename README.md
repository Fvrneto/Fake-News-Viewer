# Fake-News-Viewer

In this project we created a Fake News Detector in Python for Techlabs Düsseldorf. To accomplish this we have used natural language processing techniques and machine learning algorithms to classify articles into reliable and unreliable using scikit libraries from Python.

# Technologies

This Project is created with:

* Python 3.7 


# Description of Dataset 

The data source used for this project is the Fake news dataset from kaggle (https://www.kaggle.com/c/fake-news/data). It contains three files with .csv format for submit, test and train. Below is some description about the data files used in the project.

The original dataset ccontained five attributes for train and test as follows:

1. id:      unique id for a news article
2. title:   the title of a news article
3. author:  author of the news article
4. text:    the text of the article
5. label:   marks the article as reliable or unreliale
   - 1 = unreliable 
   - 0 = reliable

# Setup & General Information

To run the code you need to have the following packages installed:

* Matplotlib
* Natural Language Tooltik (NLTK)
   * from nltk import word_tokenize
   * from nltk.stem import SnowballStemmer
   * from nltk import ne_chunk
* Searborn
* Sklearn
   * from sklearn.model_selection import train_test_split
   * from sklearn.feature_extraction.text import TfidfVectorizer
   * from sklearn.naive_bayes import MultinomialNB
   * from sklearn.metrics import accuracy_score
   * from sklearn.metrics import precision_score
   * from sklearn.metrics import recall_score
   * from sklearn.metrics import f1_score
   * from sklearn.linear_model import LogisticRegression
   * from sklearn.ensemble import RandomForestClassifier
   * from sklearn.tree import DecisionTreeClassifier
   * from sklearn.ensemble import AdaBoostClassifier 
 * Re 

After installing the packages you need to fill in path_user = path where the main folder is located.

We split the code into two files: Fake-News-Viewer.py, that give an visual overview of the dataset and compare it the user input & Fake-News-Machine-learning.py where an algorithm is trained that can predict if the text provided by the user is reliable or unreliable compared to the dataset.

There are already two exemples of user texts in the User_text folder that can be used to see by the codes. In case you want to try with another text data, it is necessary to assing the path of this text data in path_user_data.
