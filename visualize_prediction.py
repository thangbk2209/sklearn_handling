# import plotly.offline as py
import sys
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle as pk
import numpy as np
from pandas import read_csv
from sklearn.cluster import KMeans

def visualize(vectors, labels):
    number_of_vecs = len(vectors)
    # embeddings_file = sys.argv[1]
    # wv, vocabulary = load_embeddings(embeddings_file)
    all_vec = []
    for i in range(number_of_vecs):
        # print (vectors[i])
        all_vec.append(i)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vectors[:number_of_vecs,:])

    plt.scatter(Y[:, 0], Y[:, 1],c=labels, cmap='rainbow')

    for label, x, y in zip(all_vec, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 