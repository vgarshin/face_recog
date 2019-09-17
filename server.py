import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.models import load_model
import cv2
import dlib
import utils
import pickle
import json
import glob
import skimage.transform as tr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
from cnn import get_model, load_weights
from process import ProcessFaces
IMG_SIZE = 96
print('available devices: ', [x.name for x in K.tensorflow_backend.device_lib.list_local_devices()])
from flask import Flask, request, Response

#---all functions needed---
def image_to_embedding(image, model):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2)) / 255., decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding
def find_identity(img, database, model, level=.95):
    min_dist = 100.
    identity = None
    img_embedding = image_to_embedding(img, model)
    for (label, data) in database.items():
        dist = np.linalg.norm(data[4] - img_embedding)
        if dist < min_dist:
            min_dist = dist
            identity = data[0]
    print(min_dist)
    if min_dist < level:
        return identity
    else:
        return None
def find_faces_img(img, database, model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    font_scale = max(img.shape[0], img.shape[0]) / 500
    line_scale = round(2 * font_scale)
    shift_x = 5
    shift_y = int(shift_x * font_scale)
    faces = ProcessFaces().get_faces_rects(img, 1)
    for face_rect in faces:
        img_face = ProcessFaces().align_face(img, face_rect)
        (x, y, w, h) = ProcessFaces().get_face_xywh(face_rect)
        print('found face in rect: ', (x, y, w, h))
        identity = find_identity(img_face, database, model)
        print(identity)
        if identity is not None:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, line_scale)
            cv2.putText(img, str(identity), (x1 + shift_x, y1 - shift_y), font, font_scale, color, line_scale)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def update_database(database, model, path='./update_photos/*'):
    label = 900000
    for file in tqdm(glob.glob(path), total=len(glob.glob(path))):
        label += 1 
        name = os.path.splitext(os.path.basename(file))[0]
        department, subject = 'Updated', 'Updated'
        upd_img = cv2.imread(file, 1)
        faces = ProcessFaces().get_faces_rects(upd_img, 1)
        for face_rect in faces:
            img_face = ProcessFaces().align_face(upd_img, face_rect)
            img_face = img_face.astype(int)
            database[label] = (name, department, subject, file, image_to_embedding(img_face, model))
    return database

#---load all models and data---
MODEL = load_model('./data/model.h5', custom_objects={'tf': tf})
graph = tf.get_default_graph()
print('model loaded...')
with open('./data/database.pkl', 'rb') as file:
    database = pickle.load(file)
print('database created...')
DATABASE_CUT = {k: v for k, v in database.items() if v[2] == 'Цифровая трансформация'}
DATABASE_CUT.update({k: v for k, v in database.items() if v[1] == 'Прочие'})
DATABASE_CUT = update_database(DATABASE_CUT, MODEL, path='./update_photos/*')
add_people = [
    '183555', 
    '197646', 
    '106799', 
]
DATABASE_CUT.update({k: v for k, v in database.items() if k in add_people})
del database
print('database updated...')

#---start flask---
app = Flask(__name__)
@app.route('/services/dashboard-4-9', methods=['POST'])
def get_identity():
    img_array = np.fromstring(request.get_data(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    try:
        global graph;
        with graph.as_default():
            img_proc = find_faces_img(img, DATABASE_CUT, MODEL)
    except BaseException as e:
        print(e)
        print('failed to get identity')
        response = {'message': 'face detection failed'}
        return Response(response=json.dumps(response), status=500, mimetype='application/json')
    _, data_send = cv2.imencode('.png', img_proc)
    return Response(response=data_send.tostring(), status=200, mimetype='image/png')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=21009, debug=True, use_reloader=False)