import numpy as np
import statistics as st
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd
from datetime import datetime
import scipy.stats as stats

data = pd.read_csv('../Data/SCDB_Legacy_01_justiceCentered_Citation.csv', encoding="ISO-8859-1", nrows=10000)

disp = data['caseDisposition'] # saida
out = []

for case in disp:
    if case == 2.0:
        out.append(0) #affirmative
    elif case == 3.0 or case == 4.0:
        out.append(1) #reverse
    else:
        out.append(2) #Other

# drop irrelevant columns
data = data.drop(['caseDisposition',
                  'caseId',
                  'docketId',
                  'caseIssuesId',
                  'voteId',
                  'caseName',
                  'usCite',
                  'sctCite',
                  'ledCite'],
                 axis=1)

# drop columns with more than 25% of NaN
data = data.loc[:, (data.isnull().sum() <= len(data) * 0.5)]

# verifies if column has strings
def has_no_strings (col):
    for row in col:
        if type(row) is str:
            print('"{}" is string'.format(row))
            return False
    return True

# where the program finds only numbers, input mean on NaN values
for col in data:
    print(col)
    if has_no_strings(data[col]):
        print('non-string\n')
        data[col].fillna((data[col].mean()), inplace=True)

def vetorizar_texto(cell, mapa):   #transformar palavras em vetores
    vetor = [0] * len(mapa)
    if cell in mapa:
        posicao = mapa[cell]
        # print(posicao)
        vetor[posicao] = 1
    return vetor

def vetoriza_string (col):
    dicionario = []
    for cell in col:  #colocar cada palavra encontrada no conjunto
        if (cell not in dicionario and
            cell != 'NULL' and
            cell != 'unknown' and
            cell != np.nan and
            cell != 'unidentifiable'):    # Criando um conjunto sem repetições
            dicionario.append(cell)
    total = len(dicionario)   #salvar o número de palavras catalogadas no conjunto
    # print(dicionario)
    # print(total)
    tuplas = zip(dicionario, range(total))    #dar um índice a cada palavra encontrada
    mapa = {palavra:indice for palavra, indice in tuplas}   #criar um DICIONARIO capaz de retornar o índice de determinada palavra

    vcol = []
    for cell in col:
        vcol.append(vetorizar_texto(cell, mapa))
    return vcol

def vetoriza_data (col):
    datelist = []
    for cell in col:
        try:
            dt = datetime.strptime(cell, '%m/%d/%Y')
            day = dt.strftime('%j')
            datelist.append([day, dt.year])
        except:
            datelist.append([-1, -1])
    return datelist

chiefs = np.array(vetoriza_string(data['chief']))
justice = np.array(vetoriza_string(data['justiceName']))
lexis = np.array(vetoriza_string(data['lexisCite']))
dtDecision = np.array(vetoriza_data(data['dateDecision']))
dtArgument = np.array(vetoriza_data(data['dateArgument']))
# dtRearg = np.array(vetoriza_data(data['dateRearg']))
# minor = np.array(vetoriza_string(data['lawMinor']))

def treinarePrever(nome, modelo, treino_dados, treino_marcacoes):
    k=10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

data = data.drop(['chief',
                  'justiceName',
                  'lexisCite',
                  'dateDecision',
                  'dateArgument',
                  #'dateRearg',
                  #'lawMinor'
                  ],
                 axis=1)
# print(data)
X = np.array(data)
X = np.concatenate((X,
                    chiefs,
                    justice,
                    lexis,
                    dtDecision,
                    dtArgument,
                    #dtRearg,
                    #minor
                    ),
                   axis=1)
Y = np.array(out)

X = X.astype(float)



#    for row in col:
porcentagem_de_treino = 0.8
tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

modelo = linear_model.SGDClassifier()
resultadoModelo = treinarePrever("SGDClassifier", modelo, treino_dados, treino_marcacoes)
