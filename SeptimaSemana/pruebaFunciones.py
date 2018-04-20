import numpy as np
import cv2
import FlujoOptico
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)
# Se leen las imágenes y se convierten a escala de grises
img1 = cv2.imread('img0013.jpg')
img1_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('img0014.jpg')
img2_0 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
"""
# Se calcula la imagen resultante de aplicar el algoritmo
res = FlujoOptico.calcula_imagen_resultante(img1_0, img2_0, 1, 1, True)
np.save('campoVectorial2', res[1])
img3 = res[0]
# Se normaliza el arreglo resultante
print(img3)
img3 = FlujoOptico.normaliza_arreglo(img3)
print("---------------------------")
print(img3)
# Se muestra la imagen resultante
cv2.namedWindow("0.jpg", cv2.WINDOW_NORMAL)
cv2.imshow("0.jpg", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
color = ('b','g','r')
for i,col in enumerate(color):
    histr,bins = np.histogram(img3.ravel(),256,[0,256])# histr = cv2.calcHist([img3.astype(np.uint8)], [i], None, [256], [0, 256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()

"""
# Para lo referente al campo vectorial
img = np.load('campoVectorial.npy')
img = FlujoOptico.dibuja_campo_vectorial(img)
cv2.namedWindow("0.jpg", cv2.WINDOW_NORMAL)
cv2.imshow("0.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Probar con distintos valores de ventana y dx, dy
# Hacer el campo vectorial haciendo esa cosa en pixeles pero espaciando (cada 20, 30 pixeles para no amontonar)