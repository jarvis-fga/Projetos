import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import metrics, preprocessing
from sklearn.preprocessing import Imputer
from sklearn.semi_supervised import LabelPropagation
from scipy.stats import pearsonr
import numpy as np

data = pd.read_csv('data.csv') #Ler os arquivos
print(data) #checar resultado da função
data = data.drop('communityname string', 1)

columns_quantity = len(data.columns)
knowledge_percentual = [0] * columns_quantity

x_pearson_c = []
y_pearson_c = []
pearson = []

i=-1
for col in data: #Para cada coluna em data
    i+=1          #Tenho certeza que existe uma maneira mais inteligente de fazer isso
    q=0
    for row in data[col]: #pra cada linha da coluna
        q+=1
        if row != '?':  #se a informação NÃO for desconhecida
            knowledge_percentual[i] += 1 #adicione 1 ao knowledge_percentual de índice correspondente à coluna
            #x_pearson_c.append(row)
            #y_pearson_c.append(data.loc[q-1]['ViolentCrime'])
    knowledge_percentual[i] /= q #depois divida pelo total de linhas para obter o real percentual
    #pearson.append(pearsonr(np.array(x_pearson_c).astype(np.float), np.array(y_pearson_c).astype(np.float)))
    #print("{} KP {} CP {}".format(col, knowledge_percentual[i], pearson[i][0]))
    #del x_pearson_c[:]
    #del y_pearson_c[:]


list_of_valid_cols = [] #Criando uma lista das colunas válidas
invalid_cols = [] #Criando lista das colunas excluidas
invalid_knowledge = [] #Criando lista para salvar o percentual de conhecimento sobre as listas excluidas

i=-1 #Mais uma vez, certeza que estou fazendo noobagem
for col in data: #Pra cada coluna em data
    i+=1
    if knowledge_percentual[i] >= 1:# and abs(pearson[i][0]) > 0.3 : #Se soubermos mais que x% das informações
        list_of_valid_cols.append(col) #adicione à lista de colunas válidas


list_of_valid_cols.remove('ViolentCrime')
# print(list_of_valid_cols) #print para teste

treated_dataset = pd.read_csv('data.csv', usecols=list_of_valid_cols) #criar um dataset tratado

x = np.array(treated_dataset)
y = np.array(data['ViolentCrime'])

# print(np.cov(x, y=y, rowvar=False))

# print(treated_dataset)

fit_percent = 0.8
fit_len = int(fit_percent * len(y))
test_len = len(y) - fit_len

fit_data = x[0:fit_len]
fit_label = y[0:fit_len]

val_data = x[fit_len:]
val_label = y[fit_len:]

modelLR = linear_model.LinearRegression()
modelLR.fit(fit_data, fit_label)
assertionsLR = modelLR.predict(val_data)
print('LinearRegression: ')
print(mean_squared_error(val_label, assertionsLR))

modelLasso = linear_model.Lasso()
modelLasso.fit(fit_data, fit_label)
assertions_lasso = modelLasso.predict(val_data)
print('Lasso: ')
print(mean_squared_error(val_label, assertions_lasso))

data2 = data.drop('ViolentCrime', 1)
data2 = data2.replace('?', np.NaN)
imp = Imputer()
imp.fit(data2)
treated_dataset = imp.transform(data2)
print(treated_dataset)

x = np.array(treated_dataset)
y = np.array(data['ViolentCrime'])

# print(treated_dataset)

fit_percent = 0.8
fit_len = int(fit_percent * len(y))
test_len = len(y) - fit_len

fit_data = x[0:fit_len]
fit_label = y[0:fit_len]

val_data = x[fit_len:]
val_label = y[fit_len:]

modelLR = linear_model.LinearRegression()
modelLR.fit(fit_data, fit_label)
assertionsLR = modelLR.predict(val_data)
print('LinearRegression with propagation: ')
print(mean_squared_error(val_label, assertionsLR))

modelLasso = linear_model.Lasso()
modelLasso.fit(fit_data, fit_label)
assertions_lasso = modelLasso.predict(val_data)
print('Lasso with propagation: ')
print(mean_squared_error(val_label, assertions_lasso))
