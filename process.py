import os
import cv2
import dlib
import skimage.transform as tr
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
IMG_SIZE = 96
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

class ProcessFaces:
    def __init__(self, 
                 haar_filter='./data/haarcascade_frontalface_default.xml',
                 dlib_predictor_path='./data/shape_predictor_68_face_landmarks.dat',
                 face_template_path='./data/face_template.npy'):
        self.detector = dlib.get_frontal_face_detector()
        self.face_cascade = cv2.CascadeClassifier(haar_filter)
        self.predictor = dlib.shape_predictor(dlib_predictor_path)
        self.face_template = np.load(face_template_path)
    #---align part---    
    def get_landmarks(self, img, face_rect):
        points = self.predictor(img, face_rect)
        return np.array(list(map(lambda p: [p.x, p.y], points.parts())))
    def align_face(self, img, face_rect, *, dim=96, border=0,
                   mask=INNER_EYES_AND_BOTTOM_LIP):
        landmarks = self.get_landmarks(img, face_rect)
        proper_landmarks = border + dim * self.face_template[mask]
        A = np.hstack([landmarks[mask], np.ones((3, 1))]).astype(np.float64)
        B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
        T = np.linalg.solve(A, B).T
        wrapped = tr.warp(img,
                          tr.AffineTransform(T).inverse,
                          output_shape=(dim + 2 * border, dim + 2 * border),
                          order=3,
                          mode='constant',
                          cval=0,
                          clip=True,
                          preserve_range=True)
        return wrapped
    def align_faces(self, img, face_rects, *args, **kwargs):
        result = []
        for rect in face_rects:
            result.append(self.align_face(img, rect, *args, **kwargs))
        return result
    #---detect part---    
    def get_faces_cv(self, img, scaleFactor=1.5, minNeighbors=5):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return faces
    def get_faces_rects(self, img, upscale_factor=1):
        try:
            face_rects = list(self.detector(img, upscale_factor))
        except:
            face_rects = []
        return face_rects
    def get_faces_xywh(self, face_rects):
        faces_found = []
        for face in face_rects:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            faces_found.append((x, y, w ,h))
        return faces_found
    def get_face_xywh(self, face_rect):
        x = face_rect.left()
        y = face_rect.top()
        w = face_rect.right() - x
        h = face_rect.bottom() - y
        return (x, y, w, h)
    #---process part---
    def process_all(self, 
                    path_raw='./user_photos', 
                    path_proc='./user_photos_x96'):
        labels = []
        paths = []
        if not os.path.exists(path_proc):
            os.makedirs(path_proc)
        img_files = os.listdir(path_raw)
        for img_file in tqdm(img_files):
            img_path = '{}/{}'.format(path_raw, img_file)
            img = cv2.imread(img_path)
            faces = self.get_faces_rects(img, 1)
            if faces:
                for i, face_rect in enumerate(faces):
                    try:
                        img_name = '{}/{}_{}.png'.format(path_proc, img_file[:-len('.png')], i)
                        img_face = self.align_face(img, face_rect)
                        cv2.imwrite(img_name, img_face)
                        labels.append(img_file[len('user_photo_'):][:-len('.png')])
                        paths.append(img_name)
                    except:
                        print('process failed on {}'.format(img_path))
            else:
                print('faces not found in {}'.format(img_path))
        return labels, paths
    def get_labels_paths_proc(self, path_proc='./user_photos_x96'):
        img_files = os.listdir(path_proc)
        labels = [img_file[len('user_photo_'):][:-len('_x.png')] for img_file in img_files]
        paths = ['{}/{}'.format(path_proc, img_file) for img_file in img_files]
        return labels, paths
    def get_labels_paths_raw(self, path_raw='./user_photos'):
        img_files = os.listdir(path_raw)
        labels = [img_file[len('user_photo_'):][:-len('.png')] for img_file in img_files]
        paths = ['{}/{}'.format(path_raw, img_file) for img_file in img_files]
        return labels, paths