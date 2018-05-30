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
    cam = video.create_capture("veintev2.mp4")
    ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    pw_hsv = True
    idx = 0
    flow_ant = np.zeros((234, 320, 2))


    while True:
        ret, img = cam.read()
        if img is None:
            break
        print(img.shape)
        sys.exit(-1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        arr1 = ang * 180 / np.pi / 2
        arr2 = cv.normalize(mag, None, 0, 1000, cv.NORM_MINMAX)
        flow_ant = flow
        #np.save('archivonumpy'+str(idx),flow)
        prevgray = gray
        #np.savetxt("magnitudes"+str(idx)+".txt", arr2)
        np.save("veinte/magnitudes"+str(idx), arr2)
       # np.savetxt("angulos"+str(idx)+".txt", arr1)
        np.save("veinte/angulos"+str(idx), arr1)
        draw_hsv(flow)
        #with open('archivo'+str(idx)+'idx', 'w') as outfile:
        #
        #    outfile.write('# Array shape: {0}\n'.format(flow.shape))
        #
        #    for data_slice in flow:
         #       np.savetxt(outfile, data_slice, fmt='%-7.2f')
          #      outfile.write('# New slice\n')
        #print(str(idx) +"-  "+ str((flow > 1).sum()))
        #print(flow)
        #cv.imwrite('flow'+str(idx)+".png", draw_flow(gray, flow))
        #if True:
        #    cv.imwrite('flow HSV'+str(idx)+'.png', draw_hsv(flow))
        idx += 1
        ch = cv.waitKey(5)


    cv.destroyAllWindows()


    # Para convertir mkv a mp4

    # https://askubuntu.com/questions/50433/how-to-convert-mkv-file-into-mp4-file-losslessly