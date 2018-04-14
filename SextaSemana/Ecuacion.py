# Pasar a escala de grises usando la función de opencv
import cv2
import numpy as np
import sys
import random
img1 = cv2.imread('img0008.jpg')
#img1_0 = img1[:, :, 0]
img1_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img1_1 = img1[:, :, 1]
#img1_2 = img1[:, :, 2]
img2 = cv2.imread('img0009.jpg')
#img2_0 = img2[:, :, 0]
img2_0 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img2_1 = img2[:, :, 1]
#img2_2 = img2[:, :, 2]
#img3 = cv2.imread('img0009.jpg')
img3 = np.zeros(img2_0.shape, dtype="uint32")
#img4 = np.zeros(img2_1.shape, dtype="uint32")
#img5 = np.zeros(img2_2.shape, dtype="uint32")
campo_vectorial = np.zeros((img2.shape[0], img2.shape[1], 3))
iter = 51

def normalizar(arr):
    maxi = 255
    maximo = np.max(arr) ## Equivalente a 255
    nvoarr = np.zeros(arr.shape)
    for x in range(len(arr)):
        for y in range(len(arr[x])):
            nvoarr[x][y] = int((arr[x][y] * maxi) / maximo)
    return nvoarr

def sumas(ux, uy, wx, wy, img_i, img_j):
    suma = 0
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    idx_i = 0
    idx_j = 0
    contador = 0
    suma_dx = 0
    suma_dy = 0
    for x in range(ux - wx, ux + wx):
        idx_j = 0
        for y in range(uy - wy, uy + wy):
            nva_x = x if x < img1.shape[0] else img1.shape[0] - 1
            nva_y = y if y < img1.shape[1] else img1.shape[1] - 1
            res = menor(nva_x, nva_y, arr, arr, img_i, img_j)
            suma += res[0]
            #suma += np.power(img_i[nva_x][nva_y] - img_j[xdx][ydy], 2)
            #campo_vectorial[idx_i][idx_j] = (res[1], res[2], 0 if random.randint(0,1) % 2 == 0 else 1)
            suma_dx += res[1]
            suma_dy += res[2]
            contador += 1
            idx_j += 1
        idx_i+=1
    return suma, int(suma_dx / contador), int(suma_dy / contador)


def menor(x, y, dx, dy, img_i, img_j):
    valor = np.inf
    dx_menor = 0
    dy_menor = 0
    for z, w in zip(dx, dy):
        xdx = x + z if x + z < img1.shape[0] else img1.shape[0]-1
        ydy = y + w if y + w < img1.shape[1] else img1.shape[1]-1
        temp = np.power(img_i[x][y] - img_j[xdx][ydy], 2)
        if temp < valor:
            valor = temp
            dx_menor = z
            dy_menor = w
      #  print("La dx menor es : " + str(dx_menor))
      #  print("La dy menor es : " + str(dy_menor))
    return valor, dx_menor, dy_menor


def hazFuncion(iteracion):
    for x in range(img1.shape[0]-1):
        #print( "Voy en la x : " + str(x))
        for y in range(img1.shape[1]-1):
            val = sumas(x, y, 1, 1, img1_0, img2_0)
            img3[x][y] = val[0]
            campo_vectorial[x][y] = (val[1], val[2],  0 if random.randint(0,1) % 2 == 0 else 1)
        #    img4[x][y] = sumas(x, y, 20, 20, img1_1, img2_1)
        #    img5[x][y] = sumas(x, y, 20, 20, img1_2, img2_2)

for x in range(iter):
    img3 = np.zeros(img2_0.shape, dtype="uint32")
    # img4 = np.zeros(img2_1.shape, dtype="uint32")
    # img5 = np.zeros(img2_2.shape, dtype="uint32")
    hazFuncion(x)
    for x in range(len(campo_vectorial)):
        for y in range(len(campo_vectorial[x])):
            img1_0 = cv2.line(img1_0, (x, y), (x + int(campo_vectorial[x][y][0]),
                    y + int(campo_vectorial[x][y][1])), (random.randint(0,255), random.randint(0,255), random.randint(0, 255)), 1)
    cv2.namedWindow(str(x) + "0.jpg", cv2.WINDOW_NORMAL)
    cv2.imshow(str(x) + "0.jpg", img1_0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
    img3 = np.copy(normalizar(img3))
    print("???" + str(x))
    if x % 10 == 0:
        #cv2.imwrite("nva" + str(x) + ".jpg", img1_0)
        cv2.namedWindow(str(x) + "0.jpg", cv2.WINDOW_NORMAL)
        cv2.imshow(str(x) + "0.jpg", img3)
      #  cv2.namedWindow(str(x) + "1.jpg", cv2.WINDOW_NORMAL)
      #  cv2.imshow(str(x) + "1.jpg", img4)
      #  cv2.namedWindow(str(x) + "2.jpg", cv2.WINDOW_NORMAL)
       # cv2.imshow(str(x) + "2.jpg", img5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''





