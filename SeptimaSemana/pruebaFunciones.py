import numpy as np
import cv2
import FlujoOptico

# Se leen las imágenes y se convierten a escala de grises
img1 = cv2.imread('img0008.jpg')
img1_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('img0009.jpg')
img2_0 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Se calcula la imagen resultante de aplicar el algoritmo
res = FlujoOptico.calcula_imagen_resultante(img1_0, img2_0)
img3 = res[0]
# Se normaliza el arreglo resultante
img3 = FlujoOptico.normaliza_arreglo(img3)

# Se muestra la imagen resultante
cv2.namedWindow("0.jpg", cv2.WINDOW_NORMAL)
cv2.imshow("0.jpg", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Probar con distintos valores de ventana y dx, dy
# Hacer el campo vectorial haciendo esa cosa en pixeles pero espaciando (cada 20, 30 pixeles para no amontonar)
