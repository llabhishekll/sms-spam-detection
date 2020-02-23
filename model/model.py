import pandas as pd
df = pd.read_csv('./data/data.txt',sep='\t',names=['Class','Text'])

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stopset = set(stopwords.words("english"))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=stopset,binary=True)

X = vectorizer.fit_transform(df.Text)
y = df.Class

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)

import pickle
from sklearn.externals import joblib

with open('TfidfVectorizerModel.pkl','wb') as TfidfVectorizerModel:
	pickle.dump(vectorizer.vocabulary_,TfidfVectorizerModel)

with open('AdaBoostClassifierModel.pkl','wb') as AdaBoostClassifierModel:
	joblib.dump(clf,AdaBoostClassifierModel)