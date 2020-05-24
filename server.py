from tensorflow.keras.models import load_model
from flask import Flask, request
from flask_cors import CORS
import cv2, dlib
import os, time
import numpy as np
from imutils.face_utils import shape_to_np, rect_to_bb
from constants\
  import IMAGE_SIZE, CNN_MODEL_PATH, SHAPE_PREDICTOR_PATH

app = Flask(__name__)
CORS(app)

model = load_model(CNN_MODEL_PATH)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

@app.route('/predict', methods=['POST'])
def predict():
  video = request.files['video']
  file_path = f'requested-videos/{time.time()}.webm'
  video.save(file_path)
  cap = cv2.VideoCapture(file_path)

  data = []

  while (cap.isOpened()):
    ret, frame = cap.read()
    if (ret == False): break

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = detector(frame, 0)

    if (len(faces) != 1):
      continue

    face = faces[0]
    (x, y, w, h) = rect_to_bb(face)

    landmarks = predictor(frame, face)
    landmarks = shape_to_np(landmarks)

    (lx, ly) = (landmarks[42][0], landmarks[43][1])
    (lw, lh) = (landmarks[45][0] - lx, landmarks[47][1] - ly)
    (lx, ly, lw, lh) = (lx-10, ly-10, lw+30, lh+20)

    (rx, ry) = (landmarks[36][0], landmarks[37][1])
    (rw, rh) = (landmarks[39][0] - rx, landmarks[41][1] - ry)
    (rx, ry, rw, rh) = (rx-10, ry-10, rw+30, rh+20)

    eyes = [
      frame[ly:ly+lh, lx:lx+lw],
      frame[ry:ry+rh, rx:rx+rw]
    ]

    for eye in eyes:
      eye = cv2.resize(eye, dsize=IMAGE_SIZE[::-1])
      eye = np.array(eye).reshape(*IMAGE_SIZE, 1).astype(np.float32)
      eye = eye / 255.
      data.append(eye)

  data = np.array(data)
  result = model.predict(data)
  result = list(map(lambda x: 1 if x[0] > 0.5 else 0, result))

  os.remove(file_path)
  return { 'result': result }

if __name__ == '__main__':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  app.run()
