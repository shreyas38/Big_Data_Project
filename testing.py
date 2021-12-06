#testing files
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, NGram
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SQLContext
from pyspark.sql import functions as sf
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import pickle


def access(df):
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
		if(ham_spam =='ham'):
			ham_spam_list.append(1.0)
		else:
			ham_spam_list.append(0.0)			
	
	#df.select("spam").show()
	#print(ham_spam_list)
	stopwords = text.ENGLISH_STOP_WORDS.union([""])
	tfid=TfidfVectorizer(max_features=2000,min_df=5 ,max_df=0.7,stop_words=stopwords)
	feats=tfid.fit_transform(final_sentence).toarray()
	
	#Multinomial Naive Bayers classifer
	class1='/home/pes1ug19cs467/Documents/bd/multinomialNB.sav'
	file1 = open(class1,'rb')
	classifier1 = pickle.load(file1)
	preds1=classifier1.predict(feats)
	#print("naive",preds1)
	
	acc1=accuracy_score(ham_spam_list,preds1)#calculating accuracy_score
	f1_1=f1_score(ham_spam_list,preds1,pos_label=1)#calculating F1_score
	recall1=recall_score(ham_spam_list,preds1,pos_label=1)#calculating recall
	precision1=precision_score(ham_spam_list,preds1,pos_label=1)#calculating precision
	con1=confusion_matrix(ham_spam_list,preds1,labels=[0.0,1.0])#calculating confusion matrix
	print("Multinomial Naive Bayers classifer\naccuracy : ",acc1,"\nf1_score : ",f1_1,"\nrecall : ",recall1,"\nprecision : ",precision1,"\nconfusion_matrix : \n",con1)#printing calulated results
	'''classifier1.partial_fit(feats,ham_spam_list,[0.0,1.0])
	pickle.dump(classifier1,open(class1,'wb'))
	'''
	file1.close()
	
	#SVM classifier
	class2='/home/pes1ug19cs467/Documents/bd/svm.sav'
	file2 = open(class2,'rb')
	estimator_svm = pickle.load(file2)
	preds2= estimator_svm.predict(feats)
	#print("svm",preds2)
	
	acc2=accuracy_score(ham_spam_list,preds2)#calculating accuracy_score
	f1_2=f1_score(ham_spam_list,preds2,pos_label=1)#calculating F1_score
	recall2=recall_score(ham_spam_list,preds2,pos_label=1)#calculating recall
	precision2=precision_score(ham_spam_list,preds2,pos_label=1)#calculating precision
	con2=confusion_matrix(ham_spam_list,preds2,labels=[0.0,1.0])#calculating confusion matrix
	print("\nSVM classifier\naccuracy : ",acc2,"\nf1_score : ",f1_2,"\nrecall : ",recall2,"\nprecision : ",precision2,"\nconfusion_matrix : \n",con2)#printing calulated results
	'''estimator_svm.partial_fit(feats, ham_spam_list, classes = [0.0,1.0])   #svm partial fit
	pickle.dump(estimator_svm,open(class2,'wb'))
	'''
	file2.close()
	
	#Logistic Regression
	class3='/home/pes1ug19cs467/Documents/bd/logre.sav'
	file3 = open(class3,'rb')
	estimator_log = pickle.load(file3)
	preds3=estimator_log.predict(feats)
	#print("log",preds3)
	acc3=accuracy_score(ham_spam_list,preds3)  #calculating accuracy_score
	f1_3=f1_score(ham_spam_list,preds3,pos_label=1)#calculating F1_score
	recall3=recall_score(ham_spam_list,preds3,pos_label=1)#calculating recall
	precision3=precision_score(ham_spam_list,preds3,pos_label=1)#calculating precision
	con3=confusion_matrix(ham_spam_list,preds3,labels=[0.0,1.0])#calculating confusion matrix
	print("\nLogistic Regression classifier\naccuracy : ",acc3,"\nf1_score : ",f1_3,"\nrecall : ",recall3,"\nprecision : ",precision3,"\nconfusion_matrix : \n ",con3)#printing calulated results
	'''estimator_log.partial_fit(feats, ham_spam_list, classes = [0.0,1.0])
	pickle.dump(estimator_log,open(class3,'wb'))
	'''
	file3.close()


	#Mini-batch K-means
	class4='/home/pes1ug19cs467/Documents/bd/mbk.sav'
	file4 = open(class4,'rb')
	mbk = pickle.load(file4)
	preds4=mbk.predict(feats)
	file4.close()
	acc4=accuracy_score(ham_spam_list,preds4) #calculating accuracy_score
	f1_4=f1_score(ham_spam_list,preds4,pos_label=1)#calculating F1_score
	recall4=recall_score(ham_spam_list,preds4,pos_label=1)#calculating recall
	precision4=precision_score(ham_spam_list,preds4,pos_label=1)#calculating precision
	con4=confusion_matrix(ham_spam_list,preds4,labels=[0.0,1.0])#calculating confusion matrix
	print("\nMini-Kmeans classifier\naccuracy : ",acc4,"\nf1_score : ",f1_4,"\nrecall : ",recall4,"\nprecision : ",precision4,"\nconfusion_matrix : \n ",con4,"\n\n\n")#printing calulated results
	#df = pd.DataFrame(list(zip(feats, preds4)),columns =['feats', 'preds4'])
	
	'''
	plt.bar(['Naive Bayes','SVM','Logistic Regression'],[acc1, acc2, acc3], width = 0.4)
	plt.ylabel("Accuracy")
	plt.show()
	
	plt.bar(['Naive Bayes','SVM','Logistic Regression'],[f1_1, f1_2, f1_3], width = 0.4)
	plt.ylabel("f1")
	plt.show()
	'''

	
	
	
