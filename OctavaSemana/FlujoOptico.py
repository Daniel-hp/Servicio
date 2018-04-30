import numpy as np
import cv2
import random
from sys import exit
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

    # TODO revisar que al restar el tipo de dato se preserve que 5 -6 = > -1 , no 249
    suma = 0
    for x in range(ux - wx, ux + wx):
        for y in range(uy - wy, uy + wy):
            nva_x = x if x < img_i.shape[0] else 0
            nva_y = y if y < img_i.shape[1] else 0
            xdx = x + dx if x + dx < img_i.shape[0] else 0
            ydx = y + dy if y + dy < img_i.shape[1] else 0
            prim = int(img_i[nva_x][nva_y])
            seg = int(img_j[xdx][ydx])
            res = int(prim-seg)
            res_pow = np.power(res,2)
            suma += res_pow

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
    nvoarr = (arreglo * 255) / maximo
    return nvoarr.astype("uint8")


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
    menor_dx = np.inf
    menor_dy = np.inf
    for dx in vals_dx:
        for dy in vals_dy:
            temp = calculo_epsilon(ux, uy, wx, wy, dx, dy, img_i, img_j)
            #print(str(temp) + " ... " + str(dx) +" ... " +str(dy))
          #  print(temp)
            if temp < menor:
                menor = temp
                menor_dx = dx
                menor_dy = dy
    #print(menor)
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
                val = calculo_epsilon(x, y, wx, wy, 5, 5, img_i, img_j)
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


def lee_video(video):
    """
    Método que lee un video y, posteriormente, calcula las imágenes resultantes dados
    dos frames de éste
    :param video: el nombre del video
    :return:
    """
    cap = cv2.VideoCapture(video)
    idx_frame = 0
    frame_anterior = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_actual = frame
        if idx_frame > 0:
            print("Frames: " + str(idx_frame))
            res = calcula_imagen_resultante(frame_anterior, frame_actual, 10, 10)
            np.save('ImagenResultante' + str(idx_frame), res[1])
            img3 = res[0]
            # Se normaliza el arreglo resultante
            img3 = normaliza_arreglo(img3)
            # Se muestra la imagen resultante
            #cv2.namedWindow("0.jpg", cv2.WINDOW_NORMAL)
            #cv2.imshow("0.jpg", img3)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cv2.imwrite("10x10ImagenResultante"+str(idx_frame)+".png", img3)

       # cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_anterior = frame
        idx_frame += 1
    cap.release()
    cv2.destroyAllWindows()


def calcula_piramide(img_i, img_j, wx=5, wy=5, niveles=4):
    flujo = np.zeros((img_i.shape[0], img_i.shape[1], 2), dtype="uint8")
    for nvl in range(niveles):
        print("Voy en el nivel : " + str(nvl))
        una_vez = True  # para que la operación de g se aplique una vez
        for x in range(img_i.shape[0] - 1):
            for y in range(img_i.shape[1] - 1):
                if nvl == 0 and una_vez:
                    g = np.zeros(2)
                    m = minimaliza_epsilon(x, y, wx, wy, img_i, img_j)
                    d = (m[1], m[2])
                    una_vez = False
                elif una_vez:
                    #m = minimaliza_epsilon(x, y, wx, wy, img_i, img_j)
                    d = g + d#(m[1], m[2])
                    #print(d)
                    g = 2 * (g + d)
                    una_vez = False
                #val = calculo_piramide_epsilon(x, y, wx, wy, d[0], d[1], g[0], g[1], img_i, img_j, nvl) usar misma funcion, predeterminado gx = 0, gy = 0
                if nvl == niveles-1:
                    flujo[x][y] = (d[0], d[1])

    print(flujo)
    arreglo_255 = np.ones((flujo.shape[0], flujo.shape[1], 1)) * 255
    flujo = np.concatenate((flujo, arreglo_255), axis=2)
    #print (flujo)
    return flujo


def calculo_piramide_epsilon(ux, uy, wx, wy, dx, dy, gx, gy, img_i, img_j, nivel):

    """
    En esta función se calcula la diferencia de intensidad entre vecindades centradas en los puntos (ux, uy) en la imagen I y el punto
    (ux + dx, uy + dy) en la imagen J.
    :param ux: coordenada x del punto
    :param uy: coordenada y del punto
    :param wx: distancia x de la ventana
    :param wy: distancia y de la ventana
    :param dx: distancia del punto en x
    :param dy: distancia del punto en y
    :param gx: flujo óptico gx
    :param gy: flujo óptico gy
    :param img_i: Primer imagen
    :param img_j: segunda imagen
    :return: epsilon
    """
    suma = 0
    ux = int(ux / np.power(2, nivel))
    uy = int(uy / np.power(2, nivel))
    for x in range(ux - wx, ux + wx):
        for y in range(uy - wy, uy + wy):
            # TODO revisar que al restar el tipo de dato se preserve que 5 -6 = > -1 , no 254
            nva_x = x if 0 < x < img_i.shape[0] else 0
            nva_y = y if 0 < y < img_i.shape[1] else 0
            xdx = x + gx + dx if 0 < x + gx + dx < img_i.shape[0] else 0
            ydx = y + gy + dy if 0 < y + gy + dy < img_i.shape[1] else 0
            suma += np.power(img_i[nva_x][nva_y] - img_j[int(xdx)][int(ydx)], 2)
    return suma



