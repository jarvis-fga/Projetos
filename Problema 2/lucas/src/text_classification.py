# Create by Lucas Andrade
# On 11 / 09 / 2017

import pandas as pd
import codecs
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

k = 10

file = codecs.open('../dados/comments_min.csv','rU','UTF-8')
my_data = pd.read_csv(file, sep='\t')

comments_ugly = my_data['comentarios']
marks = my_data['status']

comments = comments_ugly.str.lower().str.split(' ')
dictionary = set()

for words in comments:
	dictionary.update(words)

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

X_training = X[0:2399]
Y_training = Y[0:2399]

X_test = X[2399:2999]
Y_test = Y[2399:2999]

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
	print("{0} {1}\n".format(name, percent))
	return percent

# Multinomial NB
print("Multinomial NB is training . . .")
nb_model = MultinomialNB()
nb_model.fit(X_training, Y_training)
nb_result = nb_model.predict(X_test)
calc_percent(nb_result, "Multinomial NB: ")

# Gaussian NB
print("Gaussian NB is training . . .")
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_training, Y_training)
gaussian_nb_result = gaussian_nb.predict(X_test)
calc_percent(gaussian_nb_result, "Gaussian NB: ")

# LogisticRegression
print("Logic Regression is training . . .")
logic_regression = LogisticRegression(random_state=1)
logic_regression.fit(X_training, Y_training)
logic_regression_result = logic_regression.predict(X_test)
calc_percent(logic_regression_result, "Logic Regression: ")

# Gradient Boosting
print("Gradient Boosting is training . . .")
gradient_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)
gradient_model.fit(X_training, Y_training)
gradient_result = gradient_model.predict(X_test)
calc_percent(gradient_result, "Gradient Boosting: ")

# Random Florest !!!! Sempre um resultado diferente
print("Random Florest is training . . .")
random_f = RandomForestClassifier(random_state=1)
random_f.fit(X_training, Y_training)
random_florest_result = random_f.predict(X_test)
calc_percent(random_florest_result, "Random Florest: ")

# SVC pure
svc_pure = SVC(kernel="linear", C=1.0, random_state=0)
print("SVC Pure is training . . .")
svc_pure.fit(X_training, Y_training)
svcpure_test_result = svc_pure.predict(X_test)
calc_percent(svcpure_test_result, "SVC pure real:")

# Juntando os resultado
def get_results(comment):
    comment_a = [comment]
    print(comment_a)
    comment_b = comment_a.str.lower().str.split(' ')
    print(comment_b)
    vector_comment = comment_as_vector(comment_b, translator)
    nb = nb_model.predict(vector_comment)
    logic = logic_regression.predict(vector_comment)
    svc = svc_pure.predict(vector_comment)
    gradient = gradient_model.predict(vector_comment)
    florest = random_f.predict(vector_comment)
    results = nb + logic + svc + gradient + florest
    return results

# Usando o resultado do melhores cinco modelos
def final_result(results):
    i=0
    for result in results:
        if result < 3:
            results[i] = 0
        else:
            results[i] = 1
        i = i + 1
    calc_percent(results, "Resultado Final: ")

all_results = nb_result+logic_regression_result+svcpure_test_result+gradient_result+random_florest_result
final_result(all_results)

# def new_comment():
#     comment = "not null"
#     while(comment != ""):
#         comment = input("Type here your comment")
#         final_result(comment)
#
# new_comment()

# SVC with cross validation k = 10
# svc = svm.SVC(kernel='linear', C=1.0, random_state=0)
# print("SVC with cross validation is training . . .")
# accuracy_svc = cross_val_score(svc, X, Y, cv=k, scoring='accuracy').mean()
# print("SVC with cross val training: ", accuracy_svc)
# print("\n")
#
# # Gradient Boosting
# from sklearn.ensemble import GradientBoostingClassifier
# gradient_boosting = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
# print("Gradient Boosting is training . . .")
# accuracy_boost = cross_val_score(gradient_boosting, X, Y, cv=k, scoring='accuracy').mean()
# print("Boosting training: ", accuracy_boost)