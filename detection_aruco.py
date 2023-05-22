import cv2
from cv2 import aruco
import numpy as np 
import os 
"""
    PASO 1: Calibracion de la camara 
    PASO 2:  Detectar ArUcos.
      Primero: Se define el diccionario con el cual trabajar, asi mismo se crea los parametros para detectar el objeto
      Segundo: La imagen original se convierte en escala de grises
      la imagen es utilizada en el detector, asi como el diccionario definido en el inicio y los parametros a utilizar 
          La funcion DETECTMARKERS regresa las coordenadas de cada uno de los cuadrados identificados, tambiien el
            identificador que le corresponde a cada uno y los puntos que podrian ser, pero realmenete no lo son  
      Tercero: En el frame original, es decir, la imagen original marca en donde se encuentran los ArUcos y el id que 
          le corresponde
"""

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
cbrow = 9
cbcol = 7
#(9, 7)
path='/home/denvars/python_projects/aruco /calibracion2/'
cap = cv2.VideoCapture(0)

def img_list(path):
    list_images = os.listdir(path)
    list_image_path =[]
    for images in list_images:
        path_images = path + images
        list_image_path.append(path_images)
    return list_image_path


def calibracion(list_image_path):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((cbcol *cbrow , 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
    objpoints = []
    imgpoints = [] 

    for fname in list_image_path:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,7),None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (9,7), corners2,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs



def aruco_identify(frame, dict_aruco, parameters):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    maker_corners, maker_ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), maker_corners, maker_ids)
    return frame_markers, mak
  
 
try:
    list_image_path = img_list(path)
    ret, mtx, dist, rvecs, tvecs = calibracion(list_image_path)
    print(mtx)
    u, v = 1, 1
    while True:
        ret, frame = cap.read()
        
        frame_markers, maker_corners, maker_ids = aruco_identify(frame, dict_aruco, parameters)
        if np.all(maker_ids != None):
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(maker_corners, 0.9, mtx, dist)
            for i in range(0, maker_ids.size):
                cv2.drawFrameAxes(frame_markers, mtx, dist, rvec[i], tvec[i], 0.3)

                cv2.imshow('frame', frame_markers)
        cv2.imshow('frame', frame_markers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('frame')
    cap.release()

except KeyboardInterrupt:   
    cv2.destroyWindow('frame')
    cap.release()
