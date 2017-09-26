#!-*- coding: utf8 -*-

import pandas as pd
from sklearn.model_selection import cross_val_score
from collections import Counter
import numpy as np

classificacoes = pd.read_csv('all.csv', sep='\t') #Leia o arquivo e separe as tabulações
comentarios = classificacoes['comments'] #pegar a coluna de comentarios (titulada message)
palavras = comentarios.str.lower().str.split(' ') #jogar todas as letras para minúsculo e quebrá-las em palavras
dicionario = set()  #criar um CONJUNTO (não permite repetição)
for lista in palavras:  #colocar cada palabra encontrada no conjunto
    dicionario.update(lista)

totalDePalavras = len(dicionario)   #salvar o número de palavras catalogadas no conjunto
print(totalDePalavras)

tuplas = zip(dicionario, range(totalDePalavras))    #dar um índice a cada palavra encontrada
mapa = {palavra:indice for palavra, indice in tuplas}   #criar um DICIONARIO capaz de retornar o índice de determinada palavra

def vetorizar_texto(texto, mapa):   #transformar palavras em vetores
    vetor = [0] * len(mapa)         #o vetor terá len(mapa) números
    for palavra in texto:
        if palavra in mapa:         # e para cada ocorrência de determinada palavra
            posicao = mapa[palavra] #incrementar a posição do array correspondente
            vetor[posicao] += 1
    return vetor

vetoresdeTexto = [vetorizar_texto(texto, mapa) for texto in palavras] #fazer isso para cada um dos comentários

X = np.array(vetoresdeTexto)    #Usar algoritmos de classificação como se faz com outros casos
Y = np.array(classificacoes['status'].tolist())

porcentagem_de_treino = 0.8
tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

def treinarePrever(nome, modelo, treino_dados, treino_marcacoes):
    k=10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes)
    taxa_de_acerto = np.mean(scores)

    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = treinarePrever("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = treinarePrever("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = treinarePrever("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = treinarePrever("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print (resultados)
vencedor = resultados[max(resultados)]
print ("Vencedor: {}".format(vencedor))

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
print("Taxa de acerto do vencedor: {}".format(taxa_de_acerto))
