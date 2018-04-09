import cv2
import numpy as np
img1 = cv2.imread('img0008.jpg')
img2 = cv2.imread('img0009.jpg')
#img3 = cv2.imread('img0009.jpg')
img3 = np.zeros(img1.shape)
iter = 51


def sumas(ux, uy, wx, wy, dx, dy, img_i, img_j):
    suma = 0
    x = ux - wx
    y = uy - wy
    while x < ux + wx:
        while y < uy + wy:
            xdx = x + dx if x + dx < img1.shape[0] else x
            ydy = y + dy if y + dy < img1.shape[1] else y
            suma += np.power(img_i[x][y] - img_j[xdx][ydy], 2)
            y += 1
        x += 1
    return suma


def hazFuncion(iteracion):
    for x in range(img1.shape[0]-1):
        for y in range(img1.shape[1]-1):
            img3[x][y] = sumas(x, y, 1, 1, iteracion, iteracion, img1, img2)


for x in range(iter):
    img3 = np.zeros(img1.shape)
    hazFuncion(x)
    if x % 10 == 0:
        #cv2.imwrite("s"+str(x)+"xy.jpg", img3)
        cv2.namedWindow(str(x) + "dd.jpg", cv2.WINDOW_NORMAL)
        cv2.imshow(str(x) + "dd.jpg", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






