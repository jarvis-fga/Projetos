import csv

def carregar_acessos(arquivo_nome):
	dados = []
	marcacoes = []

	arquivo = open(arquivo_nome, 'rb')
	leitor = csv.reader(arquivo)
	leitor.next()
	for P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,Origem in leitor:
		dados.append([float(P1), float(P2), float(P3), float(P4), float(P5), float(P6), float(P7), float(P8), float(P9), float(P10), float(P11), float(P12), float(P13)])
		marcacoes.append(Origem)

	return dados, marcacoes

def taxa_acerto(resultado, gabarito):
	i=0
	acertos=0
	for r in resultado:
		if r == gabarito[i]:
			acertos=acertos+1

	taxa = 100.0*acertos/len(resultado)
	return taxa

dados, marcacoes = carregar_acessos('dados_tratados.csv')
teste, marcacoes_teste = carregar_acessos('dados_teste.csv')

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

resultado1 = modelo.predict(teste)
taxa_final = taxa_acerto(resultado1, marcacoes_teste)
print("Taxa de acerto em % :")
print(taxa_final)

