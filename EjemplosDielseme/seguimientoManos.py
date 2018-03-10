# Basado en https://s-ln.in/2013/04/18/hand-tracking-and-gesture-detection-opencv/

import cv2 as cv
import numpy as np
import math

class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Regresa la distancia entre dos puntos
    def dist(self, p1):
        return math.sqrt(math.pow((self.x-p1.x),2) + math.pow(self.y-p1.y))


class funciones:
    # Esta función regresa el radio y centro de un círculo dado 3 puntos,
    # Si un círculo no se puede crear regresa un círculo con radio cero en el punto (0,0)
    @staticmethod
    def circulo_de_puntos(p1, p2, p3):
        offset = math.pow(p2.x, 2) + math.pow(p2.y, 2)
        bc = (math.pow(p1.x, 2) + math.pow(p1.y, 2) - offset) / 2.0
        cd = (offset - math.pow(p3.x, 2) - math.pow(p3.y, 2)) / 2.0
        det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y)
        TOL = 0.0000001;
        if abs(det) < TOL:
            return Punto(0, 0), 0
        idet = 1 / det
        centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet
        centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet
        radius = math.sqrt(math.pow(p2.x - centerx, 2) + math.pow(p2.y - centery, 2))
        return Punto(centerx, centery), radius

