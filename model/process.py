import pickle
from nltk.corpus import stopwords
stopset = set(stopwords.words("english"))

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vocabularyFile =  pickle.load(open('./model/TfidfVectorizerModel.pkl', "rb"))
trainedVectorizer = CountVectorizer(decode_error="replace",vocabulary=vocabularyFile)
transformer = TfidfTransformer()

from sklearn.externals import joblib
AdaBoostClassifierModel = joblib.load(open('./model/AdaBoostClassifierModel.pkl','rb'))

def Model(data):
	fitVectorizer  = trainedVectorizer.fit_transform([str(data)])
	fitTransformer = transformer.fit_transform(fitVectorizer)
	return(AdaBoostClassifierModel.predict(fitTransformer))


