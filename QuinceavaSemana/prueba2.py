#!/usr/bin/env python

'''
example to show optical flow
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''
# Basado en https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import video
import os
import errno
import sys
# Para crear directorios, por el momento nosotros los creamos así que no hay la necesidad de
# https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
np.set_printoptions(threshold=np.nan)

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

if __name__ == '__main__':
    import sys
    print(__doc__)
    nombres = ["sonar", "sonarv2", "sonarv3", "sonarv4", "sonarv5", "hielo", "hielov2", "hielov3", "hielov4", "hielov5", "hielov6"
               , "muerte", "muertev2", "muertev6", "calaverav3", "calaverav4", "calaverav5","paz", "pazv2", "pazv3", "pazv4", "pazv5", "pazv6"
               , "soldado", "soldadov2", "soldadov3", "soldadov4", "soldadov5", "sonar", "sonarv2", "sonarv3", "sonarv4", "sonarv5",
               "veinte", "veintev2", "veintev3", "veintev4", "veintev5", "veintev6"]
    for x in nombres:
        cam = video.create_capture("videos/"+str(x)+".mp4")
        ret, prev = cam.read()
        prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        pw_hsv = True
        idx = 0
        flow_ant = np.zeros((234, 320, 2))
        while True:
            ret, img = cam.read()
            if img is None:
                break
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_ant = flow
            prevgray = gray
            np.save(str(x)+"/magnitudes"+str(idx), flow[..., 0])
            draw_hsv(flow)
            idx += 1
            ch = cv.waitKey(5)


    cv.destroyAllWindows()


    # Para convertir mkv a mp4

    # https://askubuntu.com/questions/50433/how-to-convert-mkv-file-into-mp4-file-losslessly