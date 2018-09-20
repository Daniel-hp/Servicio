# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
import sys
import cv2.optflow as cv2_flow
detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27
MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def distancia(x, y):
    return
# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            # Se dibujan los centroides

            cv2.circle(image_np, (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)), 1, (0,0,255), -1)

def mapeo_arctan(a1, a2):
    if a2 == [[]]:
        Xc = 0
        Yc = 0
    else:
        Xc = 0
        for x in a2:
            Xc += x[0]
        Xc = Xc / len(a2)

        Yc = 0
        for x in a2:
            Yc += x[1]
        Yc = Yc / len(a2)

    res = np.degrees(np.arctan2(a1[1] - Yc, a1[0] - Xc))
    return int(res/30) if res > 0 else int((res + 360) / 30)


def calcula_posicion_manos(num_hands_detect, scores, boxes, im_width, im_height, fold, pure_path_1, pure_path_2):
    puntos = np.zeros((2, 2))
    coordenadas_orig = np.zeros((2,2))
    cant_total = 0
    orientacion = lambda p_1, p_2: [[mapeo_arctan(p_1, pure_path_1), mapeo_arctan(p_2, pure_path_2)]]
    for i in range(num_hands_detect):
        if (scores[i] > _score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            centroide = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
            # Se indica a qué cuadrante pertenece el centroide
            coordenadas_orig[i] = centroide
            puntos[i] = (int(centroide[0]), int(centroide[1])) # => puntos[i] => [[p1_x,p1_y], [p2_x, p2_y]]
            cant_total += 1
    if cant_total > 1: # Si se detectan dos manos
        if puntos[0][0] < puntos[0][0]:
            c1 = puntos[0]
            c2 = puntos[1]
        else:
            c1 = puntos[1]
            c2 = puntos[0]
        return orientacion(c1,c2), [c1, c2]
        
    elif cant_total == 1: # Si se detecta una mano
        return orientacion(puntos[0], [0,0]), [puntos[0], [0, 0]]
    else:
        return orientacion([0,0], [0,0]),  [[0, 0], [0, 0]]


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)
