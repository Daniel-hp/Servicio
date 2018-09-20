# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from fastdtw_ import fastdtw
import glob
import itertools
import operator

class KNN:

    # Cuando se llama al constructor
    def __init__(self, k):
        self.k = k
        self.x_train = []
        self.y_train = []

    # Función que entrena (asigna los valores)
    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    # Función que exporta los datos de KNN
    def export(self):
        np.save("x_train", self.x_train)
        np.save("y_train", self.y_train)

    # Función que predice a qué clase pertenece el vector X
    def predict(self, x):
        dists = self.compute_distances(x)
        selection = np.argsort(dists)[0:self.k]  # Se seleccionan los k-vecinos más cercanos
        y_pred = np.empty(self.k, dtype="<U20")
        for z in range(len(y_pred)):
            y_pred[z] = self.y_train[selection[z]]
        return self.most_common(y_pred)

    # Función que computa todas las distancias del vector X a los vectores dentro del 'vecindario', usando fastdtw
    def compute_distances(self, X):
        dists = [fastdtw.fastdtw(X, y)[0] for y in self.x_train]
        return np.array(dists)

    # Función que regresa el más común en una lista, en caso de haber repeticiones regresa el de menor índice
    # https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    @staticmethod
    def most_common(L):
        # get an iterable of (item, iterable) pairs
        SL = sorted((x, i) for i, x in enumerate(L))
        # print 'SL:', SL
        groups = itertools.groupby(SL, key=operator.itemgetter(0))
        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(L)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
            return count, -min_index
        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]


def main():
    pckl = pd.read_pickle("pickle/salida_archivos.pckl")
    X = pckl.iloc[:, 0].values
    y = pckl.iloc[:, 1].values
    classifier = KNN(5)
    classifier.train(X, y)
    
    for z in glob.glob("test/*.npy"):
        print("Se predice la clase : " + str(classifier.predict(np.load(z))) +
              "\nEs de la clase : " + str(z.split("/")[1].split(".")[0].split("_")[0]))


main()
