from ui import *
import json
import numpy as np
import cv2 as cv
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def conv_to_np(coefs):
    for key in coefs:
        if isinstance(coefs[key], int):
            coefs[key] = np.int(coefs[key])
        elif isinstance(coefs[key], float):
            coefs[key] = np.float(coefs[key])
        elif isinstance(coefs[key], list):
            coefs[key] = np.array(coefs[key])
    return coefs


class MainClass:
    def __init__(self, obj):

        self.obj = obj

        self.path_1 = ''
        self.path_2 = ''
        self.path_3 = ''
        self.path_4 = ''

        self.spin_1 = 1
        self.spin_2 = 1
        self.spin_3 = 1
        self.spin_4 = 1

        self.coefs_1 = None
        self.coefs_2 = None
        self.coefs_3 = None
        self.coefs_4 = None

        self.path_save = ''
        self.path_load = ''

        # self.name_seved = 'save_coefs_'


    def main_video_col(self):
        # тут надо воткнуть многопоточность
        if not self.path_1 == '':
            print('f 1')
            self.start_video_col_1()
        if not self.path_2 == '':
            print('f 2')
            self.start_video_col_2()
        if not self.path_3 == '':
            print('f 3')
            self.start_video_col_3()
        if not self.path_4 == '':
            print('f 4')
            self.start_video_col_4()
        print('=================================================')
        self.print_info()

    def start_video_col_1(self):
        self.coefs_1 = self.colibr_by_video(self.path_1, self.obj.spinBox_1.value())

    def start_video_col_2(self):
        self.coefs_2 = self.colibr_by_video(self.path_2, self.obj.spinBox_2.value())

    def start_video_col_3(self):
        self.coefs_3 = self.colibr_by_video(self.path_3, self.obj.spinBox_3.value())

    def start_video_col_4(self):
        self.coefs_4 = self.colibr_by_video(self.path_4, self.obj.spinBox_4.value())

    def colibr_by_video(self, path, jump):
        # self.path = memory.paths[0]
        ### FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS ###
        # print(self.path)
        counter = 0
        video_capture = cv.VideoCapture(path)
        chessboardSize = (7, 6)
        self.width = 2592
        self.heigh = 1944
        frameSize = (self.width, self.heigh)
        print('frame', frameSize)
        # new_size = (648,486)

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        is_read = True
        while is_read:
            # print(counter)
            counter += 1
            is_read, img = video_capture.read()
            if not counter % jump:
                print('in ', counter)
                # counter += 1

                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
                cv.imshow('img', gray)
                cv.waitKey(500)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    print('+')

                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)

                    # Draw and display the corners
                    cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            #         cv.imshow('img', img)
            #         cv.waitKey(10)
            #         else:
            #             print('-')
            #         print(counter," / 4150")
            #         break

        cv.destroyAllWindows()
        # print(objpoints, imgpoints, frameSize)
        if len(objpoints) != 0:
            ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
            # print(ret, cameraMatrix, dist, rvecs, tvecs)
            print('всё ок')
            return {'ret': ret, 'cameraMatrix': cameraMatrix, 'dist': dist, 'rvecs': None, 'tvecs': None}
        else:
            print('пустые koefs')
            return None

    def chose_path_1(self):

        try:
            self.path_1 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_1.setText(str(self.path_1))
            # print('path:' ,memory.paths[0])

        except:
            pass

    def chose_path_2(self):

        try:
            self.path_2 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_2.setText(str(self.path_2))
            # print('path:' ,memory.paths[0])

        except:
            pass

    def chose_path_3(self):

        try:
            self.path_3 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_3.setText(str(self.path_3))
            # print('path:' ,memory.paths[0])

        except:
            pass

    def chose_path_4(self):

        try:
            self.path_4 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_4.setText(str(self.path_4))
            # print('path:' ,memory.paths[0])

        except:
            pass

    def print_info(self):
        print(self.coefs_1, self.coefs_2, self.coefs_3, self.coefs_4)

    def save_coefs(self):
        # coefs = {'ret': ret, 'cameraMatrix': cameraMatrix, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
        coefs = {'coefs_1': self.coefs_1, 'coefs_2': self.coefs_2, 'coefs_3': self.coefs_3, 'coefs_4': self.coefs_4, }
        dumped = json.dumps(coefs, cls=NumpyEncoder)
        new_d = json.loads(dumped)
        # self.path_save += 'save_coef_'+datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        print(self.path_save+'/save_coef_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.json')
        with open(self.path_save+'/save_coef_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.json', 'w') as f:
            print('Открыто для сохранения')
            json.dump(new_d, f)

    def save_path(self):
        try:
            self.path_save = QtWidgets.QFileDialog.getExistingDirectory()#[0]
            self.save_coefs()
        except TypeError:
            pass

    def read_coefs(self):
        with open(self.path_load, 'r', encoding='utf-8') as f:
            readed = json.load(f)
        return conv_to_np(readed)


    def load_path(self):
        try:
            self.path_load = QtWidgets.QFileDialog.getOpenFileName()[0]
            coefs = self.read_coefs()
            self.coefs_1 = coefs['coefs_1']
            self.coefs_2 = coefs['coefs_2']
            self.coefs_3 = coefs['coefs_3']
            self.coefs_4 = coefs['coefs_4']
            print(coefs)
        except TypeError:
            pass
