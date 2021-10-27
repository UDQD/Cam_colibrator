from ui import *
import json
import numpy as np
import cv2 as cv
from datetime import datetime
import glob
import os
import threading

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


def conv_to_np(coef):
    for ind, coefs in coef.items():

        if coefs is None:
            continue
        for key in coefs:

            if isinstance(coefs[key], int):
                coefs[key] = np.int(coefs[key])
            elif isinstance(coefs[key], float):
                coefs[key] = np.float(coefs[key])
            elif isinstance(coefs[key], list):
                coefs[key] = np.array(coefs[key])
    return coef


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

        self.path_colibr_frames = ''
        self.path_colibr_videos = ''

        # self.obj.widget.setPixmap()

        # self.image = cv.imread('C:/Users/qwed1/Desktop/fold/pic_161.jpg')


        # self.name_seved = 'save_coefs_'


    def main_video_col(self):
        # тут надо воткнуть многопоточность
        if not self.path_1 == '':
            print('f 1')
            thread_1 = threading.Thread(target=self.start_video_col_1)
            thread_1.start()
        if not self.path_2 == '':
            print('f 2')
            thread_2 = threading.Thread(target=self.start_video_col_2)
            thread_2.start()
        if not self.path_3 == '':
            print('f 3')
            thread_3 = threading.Thread(target=self.start_video_col_3)
            thread_3.start()
        if not self.path_4 == '':
            print('f 4')
            thread_4 = threading.Thread(target=self.start_video_col_4)
            thread_4.start()

        print('=================================================')
        self.print_info()

    def start_video_col_1(self):
        self.coefs_1 = self.colibr_by_video(self.path_1, self.obj.spinBox_1.value(),self.obj.widget)

    def start_video_col_2(self):
        self.coefs_2 = self.colibr_by_video(self.path_2, self.obj.spinBox_2.value(),self.obj.widget_2)

    def start_video_col_3(self):
        self.coefs_3 = self.colibr_by_video(self.path_3, self.obj.spinBox_3.value(),self.obj.widget_3)

    def start_video_col_4(self):
        self.coefs_4 = self.colibr_by_video(self.path_4, self.obj.spinBox_4.value(),self.obj.widget_4)

    def colibr_by_video(self, path, jump,widget):
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
                # cv.imshow('img', gray)
                # cv.waitKey(500)

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
                image = cv.resize(img, (412, 262))
                # cv.imshow('Video', self.image)
                # cv.waitKey(1000)
                image = QtGui.QImage(image.data, image.shape[1], image.shape[0],QtGui.QImage.Format_RGB888).rgbSwapped()
                widget.setPixmap(QtGui.QPixmap.fromImage(image))

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
            print('---')
            self.path_1 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_1.setText(str(self.path_1))
            # print('path:' ,memory.paths[0])

        except TypeError:
            print('help')

    def chose_path_2(self):

        try:
            self.path_2 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_2.setText(str(self.path_2))
            # print('path:' ,memory.paths[0])

        except TypeError:
            pass

    def chose_path_3(self):

        try:
            self.path_3 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_3.setText(str(self.path_3))
            # print('path:' ,memory.paths[0])

        except TypeError:
            pass

    def chose_path_4(self):

        try:
            self.path_4 = QtWidgets.QFileDialog.getOpenFileName()[0]
            # memory.paths[0] = self.path
            self.obj.label_path_4.setText(str(self.path_4))
            # print('path:' ,memory.paths[0])

        except TypeError:
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
            print('in')
            self.path_load = QtWidgets.QFileDialog.getOpenFileName()[0]
            coefs = self.read_coefs()
            self.coefs_1 = coefs['coefs_1']
            self.coefs_2 = coefs['coefs_2']
            self.coefs_3 = coefs['coefs_3']
            self.coefs_4 = coefs['coefs_4']
            print(coefs)
        except:
            pass


    def get_path_colibr_frames(self):
        try:
            self.path_colibr_frames = QtWidgets.QFileDialog.getExistingDirectory()#[0]
            self.obj.label_path_foto.setText(str(self.path_colibr_frames))
            if not os.path.exists(self.path_colibr_frames+'/colibrate'):
                os.mkdir(self.path_colibr_frames+'/colibrate')
            # self.save_coefs()
        except TypeError:
            pass



    def calibr_frames(self):
        images_jpg = glob.glob(self.path_colibr_frames+'/*.jpg')
        images_png = glob.glob(self.path_colibr_frames + '/*.png')
        images = []
        images.extend(images_jpg)
        images.extend(images_png)
        print(images)
        value = self.obj.spinBox_foto.value()
        if value == 1:
            coefs = self.coefs_1
        elif value == 2:
            coefs = self.coefs_2
        elif value == 3:
            coefs = self.coefs_3
        elif value == 4:
            coefs = self.coefs_4
        print(value)
        counter_foto = 0
        for image in images:
            counter_foto+=1
            image = image.replace('/','\\')
            print(image)
            img = cv.imread(image)
            # print(img)
            # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # print(len(gray))
            new_image = self.colibr_foto(img,coefs)
            print('new image')
            new_path = self.path_colibr_frames+'/colibrate/' +'calib'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+str(counter_foto)+'.jpg'
            print(new_path)
            cv.imwrite(new_path, new_image)

    def colibr_foto(self,img, coefs):
        print('in colibr')


        h, w = img.shape[:2]
        print(img.shape[:2])
        # print(type(self.coefs_1['cameraMatrix']), type(self.coefs_1['dist']))

        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(coefs['cameraMatrix'], coefs['dist'], (w, h), 1, (w, h))

        # print(newCameraMatrix, roi)
        dst = cv.undistort(img, coefs['cameraMatrix'], coefs['dist'], None, newCameraMatrix)
        # print(dst)
        return dst


    def get_path_colibr_viseos(self):
        try:
            self.path_colibr_videos = QtWidgets.QFileDialog.getExistingDirectory()#[0]
            self.obj.label_path_video.setText(str(self.path_colibr_videos))
            if not os.path.exists(self.path_colibr_videos+'/colibrate'):
                os.mkdir(self.path_colibr_videos+'/colibrate')
            # self.save_coefs()
        except TypeError:
            pass


    def calibr_videos(self):
        videos_mp4 = glob.glob(self.path_colibr_videos+'/*.mp4')
        videos_avi = glob.glob(self.path_colibr_videos + '/*.avi')
        videos = []
        videos.extend(videos_mp4)
        videos.extend(videos_avi)
        print('=================')
        print(self.path_colibr_frames)
        print(videos)
        value = self.obj.spinBox_video.value()
        if value == 1:
            coefs = self.coefs_1
        elif value == 2:
            coefs = self.coefs_2
        elif value == 3:
            coefs = self.coefs_3
        elif value == 4:
            coefs = self.coefs_4

        counter_video = 0
        # counter_foto = 0
        for video in videos:
            counter_video+=1
            video_capture = cv.VideoCapture(video)
            fps = video_capture.get(cv.CAP_PROP_FPS)
            fourcc = cv.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
            name = self.path_colibr_videos+'/colibrate/'+'calib_video_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+str(counter_video)+'.avi'
            print(name)
            video_writer = cv.VideoWriter(name, fourcc, fps, (2592, 1944))
            counter_frames = 0
            is_read, img = video_capture.read()
            while is_read:
                print('+==================')
                print(counter_frames)
                counter_frames += 1

                print(is_read)
                new_frame = self.colibr_foto(img, coefs)
                print('new frame')
                video_writer.write(new_frame)
                print('write')
                # cv.imshow('frame', new_frame)
                # if cv.waitKey(1) & 0xFF == ord('q'):
                #     break
                is_read, img = video_capture.read()
                print('+==================')
            print('загрузка видео')
            video_writer.release()
            video_capture.release()



            # counter_foto+=1
            # image = image.replace('/','\\')
            # print(image)
            # img = cv.imread(image)
            # # print(img)
            # # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # # print(len(gray))
            # new_image = self.colibr_foto(img)
            # print('new image')
            # new_path = self.path_colibr_frames+'/colibrate/' +'calib'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+str(counter_foto)+'.jpg'
            # print(new_path)
            # cv.imwrite(new_path, new_image)