import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import metrics, preprocessing
from sklearn.preprocessing import Imputer
from sklearn.semi_supervised import LabelPropagation
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
            x_pearson_c.append(row)
            y_pearson_c.append(data.loc[q-1]['ViolentCrime'])
    knowledge_percentual[i] /= q #depois divida pelo total de linhas para obter o real percentual
    pearson.append(pearsonr(np.array(x_pearson_c).astype(np.float), np.array(y_pearson_c).astype(np.float)))
    print("{} KP {} CP {}".format(col, knowledge_percentual[i], pearson[i][0]))
    del x_pearson_c[:]
    del y_pearson_c[:]



def fit_p(kp, cc, data, pearson):
    list_of_valid_cols = [] #Criando uma lista das colunas válidas

    i=-1 #Mais uma vez, certeza que estou fazendo noobagem
    for col in data: #Pra cada coluna em data
        i+=1
        if knowledge_percentual[i] >= kp and abs(pearson[i][0]) >= cc : #Se soubermos mais que x% das informações
            list_of_valid_cols.append(col) #adicione à lista de colunas válidas

    list_of_valid_cols.remove('ViolentCrime')
    # print(list_of_valid_cols) #print para teste

    treated_dataset = pd.read_csv('data.csv', usecols=list_of_valid_cols) #criar um dataset tratado

    treated_dataset = treated_dataset.replace('?', np.NaN)

    if len(list_of_valid_cols) == 0:
        print('LinearRegression with propagation (kp = {}, cc = {}): '.format(kp, cc))
        print('0 valid cols')
        return -1

    imp = Imputer()
    imp.fit(treated_dataset)
    treated_dataset = imp.transform(treated_dataset)
    #print(treated_dataset)

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
    print('LinearRegression with propagation (kp = {}, cc = {}): '.format(kp, cc))
    error = mean_squared_error(val_label, assertionsLR)
    print(error)
    return error

'''
    modelLasso = linear_model.Lasso()
    modelLasso.fit(fit_data, fit_label)
    assertions_lasso = modelLasso.predict(val_data)
    print('Lasso with propagation: ')
    print(mean_squared_error(val_label, assertions_lasso))
'''
##################################################################

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
plotx = np.arange(0, 1, 0.05)
ploty = np.arange(0, 1, 0.05)

minerror = 1
maxerror = 0

z = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        z[i][j] = fit_p(plotx[i], ploty[j], data, pearson)
        if z[i][j] < minerror and z[i][j] > 0:
            minerror = z[i][j]
        if z[i][j] > maxerror and z[i][j] < 1:
            maxerror = z[i][j]
plotx, ploty = np.meshgrid(plotx, ploty)
# Plot the surface.
surf = ax.plot_surface(plotx, ploty, z, cmap=cm.coolwarm, vmin=minerror, vmax=maxerror,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(minerror, maxerror)
ax.set_zlabel("error (negative means impossibility)")
ax.set_xlabel("minimum pearson coefficient acceptable")
ax.set_ylabel("minimum knowledge percentual acceptable")
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

##########################################################


kp = 1
while kp > 0:
    kp = float(input('knowledge_percentual min: '))
    cc = float(input('pearson coefficient min: '))
    fit_p(kp, cc, data, pearson)
