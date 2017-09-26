import numpy as np

bag_of_punctuation = [".", "-", ",", "?", "!", "'", "(", ")", ":"]


def new_Word(word, positive, negative):
	word = Word(word, positive, negative)
	return word


# Caracteres
with open("../dados/all_comments.csv", 'r') as infile:
	with open("../dados/comments_min2.csv", 'w') as outfile:
		data = infile.read()
		data = data.lower()
		for c in data:
			for word in bag_of_punctuation:
				if word == c:
					data = data.replace(c, "")

		outfile.write(data)

words = []
count = 0
positive = []
negative = []
# Contando palavras
with open('../dados/comments_min2.csv', 'r') as f:
	contents = f.readlines()

for line in contents:
	for w in line.split():
		if w not in words:
			words.append(w)
			positive.append(0.0)
			negative.append(0.0)
			count = count + 1

print(len(positive))
print(len(negative))
print("Total de palavras = {0}".format(count))

# Gerando arquivos positivos e negativos
with open("../dados/comments_min2.csv", "rt") as fin:
	with open("../dados/positivos.csv", "wt") as fout:
		for line in fin:
			if "	1" in line:
				fout.write(line)

with open("../dados/comments_min2.csv", "rt") as fin:
	with open("../dados/negativos.csv", "wt") as fout:
		for line in fin:
			if "	0" in line:
				line = line.lower()
				fout.write(line)

# Relatorio das palavras

with open("../dados/positivos.csv", "rt") as filep:
	contents_positive = filep.readlines()

for linep in contents_positive:
	for w in linep.split():
		if w in words:
			i = words.index(w)
			positive[i] = positive[i] + 1

with open("../dados/negativos.csv", "rt") as filen:
	contents_negative = filen.readlines()

for linen in contents_negative:
	for w in linen.split():
		if w in words:
			i = words.index(w)
			negative[i] = negative[i] + 1

words_out = []

for word in words:
	j = words.index(word)
	total_aparicoes = positive[j] + negative[j]
	if total_aparicoes > 0:
		porcentagem = 100.0 * positive[j] / total_aparicoes
		if (porcentagem == 50):
			words_out.append(word)
		# print("A palavra {0} aparece {1} porcento das vezes".format(word, porcentagem))

with open("../dados/positivos.csv", "rt") as filep:
	contents_positive = filep.readlines()

for linep in contents_positive:
	for w in linep.split():
		if w in words:
			i = words.index(w)
			positive[i] = positive[i] + 1

# Removendo Palavras
with open("../dados/comments_min2.csv", "rt") as filef:
	contents_reduce = filef.readlines()

with open("../dados/comments_min.csv", "wt") as fout:
	for line in contents_reduce:
		new_line = line
		for w in new_line.split():
			if w in words_out:
				new_line = new_line.replace(w, '')
		fout.write(new_line)
