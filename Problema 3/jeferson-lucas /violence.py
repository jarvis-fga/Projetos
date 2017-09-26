import sklearn
import numpy
import pandas
import matplotlib.pyplot as plot

def read_labels():
    labels = []
    file = open('data/variables_names.txt', 'r')
    for line in file:
        line = line[:-1]
        labels.append(line)
    return labels

#print(labels)

communities = pandas.read_csv('data/communities_data.txt', sep=",", names=read_labels())
print(communities)
#communities.to_csv('data/communities.csv', index=False, encoding='utf-8')