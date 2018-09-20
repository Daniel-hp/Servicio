

from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
import sys

"""
https://github.com/victordibia/handtracking
"""

cap = cv2.VideoCapture(0)
detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.1, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=320, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=180, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=0, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    parser.add_argument('-video', dest='video', help="Nombre del video", default="hielo", type=str)
    parser.add_argument('-salida', dest='salida', help="Folder donde se guardarán los resultados", default=0, type=int)
    args = parser.parse_args()
    if args.salida == 0:
       folder = "train"
    else:
       folder = "test"
    for video in args.video.split(","):
        cap = cv2.VideoCapture("videos/" + str(video) + ".mp4")
        print("cargando: " + str(video))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        start_time = datetime.datetime.now()
        num_frames = 0
        im_width, im_height = (500, 500) #(cap.get(3), cap.get(4))
        scores_ant = None
        boxes_ant = None
        # max number of hands we want to detect/track
        arreglo_detecciones = np.array([[0,0]])
        arreglo_detecciones_2 = np.array([0,0])
        centroide__anterior = [[0, 0], [0, 0]]
        pure_path_1 = [[0,0]]
        pure_path_2 = [[0,0]]
        num_hands_detect = 2
        if args.display > 0:
            cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

        while True:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            ret, image_np = cap.read()
            if ret == False:
                break
            #image_np = cv2.flip(image_np, 1)
            image_np = cv2.resize(image_np, (500,500))
            try:
                next = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # Frame para flujo óptico
                #next = image_np
                img_orig = image_np
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # actual detection
            boxes, scores = detector_utils.detect_objects(
                image_np, detection_graph, sess)

            # draw bounding boxes
            detector_utils.draw_box_on_image(
                num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

            detec, centroide__anterior = detector_utils.calcula_posicion_manos(num_hands_detect, scores, boxes, im_width, im_height, video, pure_path_1, pure_path_2)
            if not np.array_equal(centroide__anterior[0],pure_path_1[-1]): # Si se detecta movimiento se agrega al camino
                if np.array_equal([0.0,0.0], centroide__anterior[0]): # Si el movimiento que se detecta es ir a (0,0) se asume hubo problemas en la detección y se vuelve a insertar el último elemento
                    pure_path_1 = np.insert(pure_path_1, len(pure_path_1), pure_path_1[-1], 0)
                else:
                    pure_path_1 = np.insert(pure_path_1, len(pure_path_1), centroide__anterior[0], 0)
            if not np.array_equal(centroide__anterior[1] ,pure_path_2[-1]): # Si se detecta movimiento se agrega al camino
                if np.array_equal([0.0,0.0], centroide__anterior[1]):
                    pure_path_2 = np.insert(pure_path_2, len(pure_path_2), pure_path_2[-1], 0)
                else:
                    pure_path_2 = np.insert(pure_path_2, len(pure_path_2), centroide__anterior[1], 0)
            arreglo_detecciones = np.concatenate((arreglo_detecciones, [[detec[0][0], detec[0][1]]]))
            prev_frame = next
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time
            scores_ant = scores
            boxes_ant = boxes
            if (args.display > 0):
                # Display FPS on frame
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image(
                        "FPS : " + str(int(fps)), image_np)

                cv2.imshow('Single-Threaded Detection', cv2.cvtColor(
                    image_np, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                print("frames processed: ",  num_frames,
                      "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
        arreglo_detecciones = np.delete(arreglo_detecciones, 0, axis=0) # Se elimina el primer elemento, dado que no tiene utilidad
        np.save(str(folder)+"/"+str(video.split(".")[0]) + "_detecambas", arreglo_detecciones)
