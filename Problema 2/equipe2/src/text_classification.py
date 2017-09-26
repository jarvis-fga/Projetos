# Create by Lucas Andrade
# On 11 / 09 / 2017

import pandas as pd 
import codecs
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
k = 10

file = codecs.open('../all_comments_reduce.csv','rU','UTF-8')
my_data = pd.read_csv(file, sep='\t')

comments_ugly = my_data['comments']
marks = my_data['status']

comments = comments_ugly.str.lower().str.split(' ')
dictionary = set()

for words in comments:
	dictionary.update(words)

print len(dictionary)
tuples = zip(dictionary, xrange(len(dictionary)))
translator = { word:number for word,number in tuples}

def comment_as_vector(comment, translator):
	vector = [0] * len(dictionary)
	for word in comment:
		if word in translator:
			position = translator[word]
			vector[position] += 1

	return vector

def vector_all_comments(comments, translator):
	new_comments = [comment_as_vector(comment, translator) for comment in comments]
	return new_comments

X = vector_all_comments(comments, translator)
Y = list(marks)

X_training = X[0:799]
Y_training = Y[0:799]

X_test = X[799:999]
Y_test = Y[799:999]

def check_corrects(predicted):
	accerts = predicted - Y_test
	total_accerts = 0
	for accert in accerts:
		if accert == 0:
			total_accerts+=1
	return total_accerts

def calc_percent(predicted, name):
	accerts = check_corrects(predicted)
	percent = 100.0 * accerts/len(Y_test)
	print("{0} {1}".format(name, percent))
	return percent

# SVC pure
svc_pure = SVC()
print("SVC Pure is training . . .")
svc_pure.fit(X_training, Y_training)
svcpure_test_result = svc_pure.predict(X_test)
calc_percent(svcpure_test_result, "SVC pure real:")
print()

# Implement poly SVC 
poly_svc = svm.SVC(kernel='poly', degree=3, C=1.0)
accuracy_poly_svc = cross_val_score(poly_svc, X_training, Y_training, cv=k, scoring='accuracy')
print("poly_svc: ", accuracy_poly_svc.mean())
print("")

svc = svm.SVC(kernel='linear', C=1.0)
print("SVM is training . . .")
accuracy_svc = cross_val_score(svc, X_training, Y_training, cv=k, scoring='accuracy').mean()
print("SVM Linear training: ", accuracy_svc)
print("")

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
print("Gradient Boosting is training . . .")
accuracy_boost = cross_val_score(gradient_boosting, X, Y, cv=k, scoring='accuracy').mean()
print("Boosting training: ", accuracy_boost)