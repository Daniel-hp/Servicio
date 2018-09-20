import numpy as np
import pandas as pd
import argparse

def lee_archivo(nombre):
    return np.load("train/"+ str(nombre)+ "_detecambas.npy")


def convierte_a_pickle(archivos):
    print(archivos)
    return archivos.to_pickle("pickle/salida_archivos.pckl")


def junta_archivos(n_archivos, n_clases):
    cant_archivos = len(n_archivos)
    df_resultante = pd.DataFrame(np.zeros((cant_archivos, 2)))
    df_resultante = df_resultante.astype('object')
    idx = 0
    primera_vez = True
    for x, y in zip(n_archivos, range(0,cant_archivos)):
        if y % 6 == 0 and not primera_vez:
            idx += 1
        primera_vez = False
        df_resultante[0][y] = lee_archivo(x)
        df_resultante[1][y] = n_clases[idx]


    return df_resultante



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', dest='files', type=str,
                        default="", help='Files to load')
    parser.add_argument('-n', '--names', dest='names', type=str,
                        default="", help='Names of classes')
    args = parser.parse_args()
    archivos = np.zeros((len(args.files.split(','))))
    nombres_archivos = args.files.split(',')
    nombres_clases = args.names.split(',')
    convierte_a_pickle(junta_archivos(nombres_archivos, nombres_clases))





if __name__ == '__main__':
    main()
