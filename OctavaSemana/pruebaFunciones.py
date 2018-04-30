import numpy as np
import cv2
import FlujoOptico
import joblib
from matplotlib import pyplot as plt
from scipy.ndimage import imread
np.set_printoptions(threshold=np.nan)

# Se leen las imágenes y se convierten a escala de grises
img1 = cv2.imread('img0013.jpg')
img1_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('img0014.jpg')
img2_0 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
"""

# Se calcula la imagen resultante de aplicar el algoritmo
ventana = [1, 2, 3, 5]
for x in ventana:
    print("Ventana: " + str(x))
    res = FlujoOptico.calcula_imagen_resultante(img1_0, img2_0, x, x, True)
    np.save('campoVectorial2' + str(x), res[1])
    img3 = res[0]
    # Se normaliza el arreglo resultante
    img3 = FlujoOptico.normaliza_arreglo(img3)
    cv2.imwrite("MINIMIZAv2nuevosIntentos" + str(x)+".png", img3)
    # Se muestra la imagen resultante

    
#cv2.namedWindow("0.jpg", cv2.WINDOW_NORMAL)
#cv2.imshow("0.jpg", img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
"""
"""
img3 = FlujoOptico.calcula_piramide(img1_0, img2_0)
#print(img1_0[0][0])
#print(img2_0[1][1])
#print(img1_0[0][1])
#print(img2_0[1][2])
#print(img1_0[0][-1])
#print(img2_0[1][0])
#print(img1_0[1][0])
#print(img2_0[2][1])
#print(img1_0[1][1])
#print(img2_0[2][2])
#print(img1_0[1][-1])
#print(img2_0[2][0])
#print(img1_0[-1][0])
#print(img2_0[0][1])
#print(img1_0[-1][1])
#print(img2_0[0][2])
#print(img1_0[-1][-1])
#print(img2_0[0][0])
#print(FlujoOptico.calculo_epsilon(0,0,1,1,1,1,img1_0,img2_0))
cv2.imwrite("Piramide.png", img3)

color = ('b','g','r')
for i,col in enumerate(color):
    histr,bins = np.histogram(img3.ravel(),256,[0,256])# histr = cv2.calcHist([img3.astype(np.uint8)], [i], None, [256], [0, 256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()

"""
"""
# Para lo referente al campo vectorial
img = np.load('ImagenResultante 1.npy')
img = FlujoOptico.dibuja_campo_vectorial(img)
cv2.namedWindow("0.jpg", cv2.WINDOW_NORMAL)
cv2.imshow("0.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img3 = FlujoOptico.calcula_piramide(img1_0, img2_0)
cv2.imwrite("Piramide2.png", img3)
#
#FlujoOptico.lee_video("veinte.mp4")
# Probar con distintos valores de ventana y dx, dy
# Hacer el campo vectorial haciendo esa cosa en pixeles pero espaciando (cada 20, 30 pixeles para no amontonar)