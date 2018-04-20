import numpy as np
import cv2
import random
def calculo_epsilon(ux, uy, wx, wy, dx, dy, img_i, img_j):

    """
    En esta función se calcula la diferencia de intensidad entre vecindades centradas en los puntos (ux, uy) en la imagen I y el punto
    (ux + dx, uy + dy) en la imagen J.
    :param ux: coordenada x del punto
    :param uy: coordenada y del punto
    :param wx: distancia x de la ventana
    :param wy: distancia y de la ventana
    :param dx: distancia del punto en x
    :param dy: distancia del punto en y
    :param img_i: Primer imagen
    :param img_j: segunda imagen
    :return: epsilon
    """
    suma = 0
    for x in range(ux - wx, ux + wx):
        for y in range(uy - wy, uy + wy):
            nva_x = x if x < img_i.shape[0] else 0
            nva_y = y if y < img_i.shape[1] else 0
            xdx = x + dx if x + dx < img_i.shape[0] else 0
            ydx = y + dy if y + dy < img_i.shape[1] else 0
            suma += np.power(img_i[nva_x][nva_y] - img_j[xdx][ydx], 2)
    return suma


def normaliza_arreglo(arreglo):
    """
    En esta función se normaliza el arreglo
    :param arreglo: El arreglo a normalizar
    :return: El arreglo normalizado
    """
    maxi = 255 # Equivalente a 255
    maximo = np.max(arreglo)
    if maximo == 0:
        return arreglo  # Indica todos son 0
    #nvoarr = np.zeros(arreglo.shape, dtype=arreglo.dtype)
    #for x in range(arreglo.shape[0]):
     #   for y in range(arreglo.shape[1]):
         #   temp = arreglo[x][y] / maximo
         #   valor_normalizado = maxi * temp
         #   nvoarr[x][y] = int(valor_normalizado)
    nvoarr = (arreglo * 255) / maximo
    return nvoarr


def minimaliza_epsilon(ux, uy, wx, wy, img_i, img_j):
    """
    Función que minimiza el valor de epsilon
    :param ux: coordenada x del punto
    :param uy: coordenada y del punto
    :param wx: distancia x de la ventana
    :param wy: distancia y de la ventana
    :param img_i: Primer imagen
    :param img_j: segunda imagen
    :return:
    """

    # Situación ideal donde dx <= wx, donde el punto en la imagen j se encuentra dentro de la ventana de integración
    vals_dx = np.arange(-wx, wx + 1)
    vals_dy = np.arange(-wy, wy + 1)
    menor = np.inf
    menor_dx = 0
    menor_dy = 0
    for dx in vals_dx:
        for dy in vals_dy:
            temp = calculo_epsilon(ux, uy, wx, wy, dx, dy, img_i, img_j)
            if temp < menor:
                menor = temp
                menor_dx = dx
                menor_dy = dy
    return menor, menor_dx, menor_dy


def calcula_imagen_resultante(img_i, img_j, wx=2, wy=2, minimizar=False):
    """
    Función que calcula la imagen resultante tras calcular epsilon para todos los valores de
    la imagen, ya sea de manera normal o minimizando
    :param img_i: La primer imagen
    :param img_j: La segunda imagen
    :param minimizar: El parámetro que indica si se va a minimizar la epsilon
    :return: La imagen resultante
    """
    img_resultante = np.zeros(img_i.shape)
    campo_vectorial = np.zeros((img_i.shape[0], img_i.shape[1], 3))
    for x in range(img_i.shape[0]-1):
        if x % 50 == 0:
            print("Voy en la x" + str(x))
        for y in range(img_i.shape[1]-1):
            if minimizar:
                val = minimaliza_epsilon(x, y, wx, wy, img_i, img_j)
                img_resultante[x][y] = val[0]
                campo_vectorial[x][y] = (val[1], val[2], 1)
            else:
                # Se fijan los valores de dx y dy en 1
                val = calculo_epsilon(x, y, wx, wy, 20, 20, img_i, img_j)
                img_resultante[x][y] = val
    return img_resultante, campo_vectorial


def dibuja_campo_vectorial(campo_vectorial):
    """
    Función que dibuja el campo vectorial
    :param campo_vectorial: El campo vectorial a dibujar
    :return: La imagen con el campo vectorial
    """
    imagen = np.zeros(campo_vectorial.shape)
    for x in range(len(campo_vectorial)):
        for y in range(len(campo_vectorial[x])):
            # Para blanco y negro
            """
                        arr = [0,255]
                        idx = 0 if random.randint(0,1) % 2 == 0 else 1
                        imagen = cv2.arrowedLine(imagen, (x, y), (x + int(campo_vectorial[x][y][0]),
                                                                  y + int(campo_vectorial[x][y][1])),
                                                 (arr[idx], arr[idx], arr[idx]), 3)
                        """
            # Para color
            imagen = cv2.arrowedLine(imagen, (x, y), (x + int(campo_vectorial[x][y][0]),
                                                      y + int(campo_vectorial[x][y][1])),
                                     (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254)), 3)

    return imagen



