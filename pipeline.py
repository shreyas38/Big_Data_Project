from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, NGram
from pyspark.ml import Pipeline
import numpy as np
from pyspark.sql import functions as sf
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text
from sklearn.cluster import MiniBatchKMeans
import pickle

#multinominalnb intialistaion
classifier1=MultinomialNB()

#svm intialistaion
estimator_svm = SGDClassifier(loss='hinge', penalty='l1', l1_ratio=1)

#log reg initialisation
estimator_log = SGDClassifier(loss='log', penalty='l1', l1_ratio=1)

#minikmeans initialisation
mbk = MiniBatchKMeans(init ='k-means++', n_clusters = 2, batch_size = 1500)
	

def sender(df):
	#pre-processing
	#combine title and message colummns
	df = df.withColumn('message', sf.concat(sf.col('title'),sf.lit(' '), sf.col('message')))
	#df.show()
	#dropping column title since it is combined with column message
	df=df.drop("title")
	# Extract word
	tokenizer = Tokenizer().setInputCol('message').setOutputCol('words')

	# Custom stopwords
	stopwords = StopWordsRemover().getStopWords() + ['-']

	# Remove stopwords
	remover = StopWordsRemover().setStopWords(stopwords).setInputCol('words').setOutputCol('filtered')

	pipeline_proprocess = Pipeline(stages = [tokenizer, remover])
	preprocessed = pipeline_proprocess.fit(df)
	df=preprocessed.transform(df)
	#df.show()
	
	x= df.collect()
	
	final_sentence=[]
	ham_spam_list=[]
	for i in range(len(x)):
		string=""
		words=x[i]['filtered']
		ham_spam=x[i]['spam']
		string=' '.join(words)
		#print(string)
		final_sentence.append(string)
		if(ham_spam =='ham'):# label encoding 
			ham_spam_list.append(1.0)
		else:
			ham_spam_list.append(0.0)			
	
	
	stopwords = text.ENGLISH_STOP_WORDS.union([""])
	tfid=TfidfVectorizer(max_features=2000,min_df=5 ,max_df=0.7,stop_words=stopwords) # vectorizing
	feats=tfid.fit_transform(final_sentence).toarray()

	
	
	#Multinomial Naive Bayers classifer
	class1='/home/pes1ug19cs467/Documents/bd/multinomialNB.sav'
	#file1 = open(class1,'rb')
	#classifier1 = pickle.load(file1)
	classifier1.partial_fit(feats,ham_spam_list,[0.0,1.0])
	pickle.dump(classifier1,open(class1,'wb')) #saving Model
	#file1.close()
	
	
	#SVM classifier
	class2='/home/pes1ug19cs467/Documents/bd/svm.sav'
	#file2 = open(class2,'rb')
	#estimator_svm = pickle.load(file2)
	estimator_svm.partial_fit(feats, ham_spam_list, classes = [0.0,1.0])   #svm partial fit
	pickle.dump(estimator_svm,open(class2,'wb'))#saving Model
	#file2.close()
	
	#Logistic Regression
	class3='/home/pes1ug19cs467/Documents/bd/logre.sav'
	#file3 = open(class3,'rb')
	#estimator_log = pickle.load(file3)
	estimator_log.partial_fit(feats, ham_spam_list, classes = [0.0,1.0])   #logistic regression partial fit
	pickle.dump(estimator_log,open(class3,'wb'))#saving Model
	#file3.close()
	
	#Mini-batch K-means
	class4='/home/pes1ug19cs467/Documents/bd/mbk.sav'
	#file4 = open(class4,'rb')
	#mbk = pickle.load(file4)
	mbk.partial_fit(feats)   #minikmeans partial fit
	pickle.dump(mbk,open(class4,'wb'))#saving Model
	#file4.close()

	return df
	
	
	
#def bayer(df):
	
	
