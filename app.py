import torch
import numpy as np
import cv2
import mysql.connector
from datetime import datetime

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="testpython"
)

mycursor = db.cursor()


# Printing the connection object


# res = mycursor.execute(
#     "SELECT * FROM test_table")
# print(res)

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    check, frame = cap.read()
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if 'awake' in results.pandas().xyxy[0].value_counts('name'):
        mycursor.execute(
            "INSERT INTO test_table(name, indicate, timestamp) values('JR', 'bangun', '{}')".format(datetime.now()))
        db.commit()
    if 'ngantuk' in results.pandas().xyxy[0].value_counts('name'):
        mycursor.execute(
            "INSERT INTO test_table(name, indicate, timestamp) values('JR', 'tidur', '{}')".format(datetime.now()))
        db.commit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
