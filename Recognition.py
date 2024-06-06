# import cv2
# import numpy as np
# import os
# import csv
# import time
# import pickle

# from sklearn.neighbors import KNeighborsClassifier


# from datetime import datetime


# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# with open('data/name.pkl', 'rb') as w:
#     LABELS = pickle.load(w)

# with open('data/face_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# # knn=KNeighboursClassifier(n_neighbours=5)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES,LABELS)

# imagebackground=cv2.imread("bg.png")

# COL_NAMES=['NAME''TIME']

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray,1.3,5)


#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resize_img = cv2.resize(crop_img, (50, 50))
#         output=knn.predict(resize_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#         exist=os.path.isfile("Attendance/Attendance_"+date+".csv")
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,225),1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,225),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,225),-1)
#         cv2.putText(frame,(x,y),(x+w,y+h),(50,50,225),1)
#         attendance=[str(output[0]),str(timestamp)]
#         imagebackground[162:162+480,55:55+640]=frame


#         cv2.imshow("frame",imagebackground)
#         k=cv2.waitKey(1)
#         if k==ord('o'):
#             time.sleep(5)

#             if exist:
#                 with open("Attendance/Attendance_"+date+",.csv","+a") as csvfile:
#                     writer=csv.writer(csvfile)
#                     writer.writerow(attendance)

#                 csvfile.close()

#             else:
#                 with open("Attendance/Attendance_"+date+",.csv", "+a") as csvfile:
#                     writer=csv.writer(csvfile)
#                     writer.writerow(COL_NAMES)
#                     writer.writerow(attendance)

#                 csvfile.close()

#         if k==ord('q'):
#             break

# video.release()
# cv2.destroyAllWindow()


# import cv2
# import numpy as np
# import os
# import csv
# import time
# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from datetime import datetime

# # Initialize video capture and face detector
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Load labels and face data
# with open('data/name.pkl', 'rb') as w:
#     LABELS = pickle.load(w)

# with open('data/face_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# # Convert LABELS to a numpy array
# LABELS = np.array(LABELS)

# # Print shapes of FACES and LABELS to debug the issue
# print("Shape of FACES:", FACES.shape)
# print("Shape of LABELS:", LABELS.shape)

# # Check if the lengths match
# if len(FACES) != len(LABELS):
#     raise ValueError(f"Mismatch in number of samples: FACES has {len(FACES)} samples, LABELS has {len(LABELS)} samples.")

# # Initialize KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# # Load background image
# imagebackground = cv2.imread("img.jpg")

# # Column names for CSV
# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#         output = knn.predict(resize_img)
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#         exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 225), -1)
#         cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#         attendance = [str(output[0]), str(timestamp)]
#         imagebackground[162:162+480, 55:55+640] = frame

#         cv2.imshow("frame", imagebackground)
#         k = cv2.waitKey(1)
#         if k == ord('o'):
#             time.sleep(5)
#             with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 if not exist:
#                     writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)

#         if k == ord('q'):
#             break

# video.release()
# cv2.destroyAllWindows()




# last working


# import cv2
# import numpy as np
# import os
# import csv
# import time
# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from datetime import datetime

# # Initialize video capture and face detector
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Load corrected labels and face data
# with open('data/name.pkl', 'rb') as w:
#     LABELS = pickle.load(w)

# with open('data/face_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# # Convert LABELS to a numpy array
# LABELS = np.array(LABELS)

# # Print shapes of FACES and LABELS to debug the issue
# print("Shape of FACES:", FACES.shape)
# print("Shape of LABELS:", LABELS.shape)

# # Check if the lengths match
# if len(FACES) != len(LABELS):
#     raise ValueError(f"Mismatch in number of samples: FACES has {len(FACES)} samples, LABELS has {len(LABELS)} samples.")

# # Initialize KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# # Get the dimensions of the video frame
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Load the background image and resize it to match the video frame size
# background_path = "img.jpg"  # Change this to the path of your new image
# imagebackground = cv2.imread(background_path)
# imagebackground = cv2.resize(imagebackground, (frame_width, frame_height))

# # Column names for CSV
# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#         output = knn.predict(resize_img)
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#         exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 225), -1)
#         cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#         attendance = [str(output[0]), str(timestamp)]
#         imagebackground[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

#         cv2.imshow("frame", imagebackground)
#         k = cv2.waitKey(1)
#         if k == ord('o'):
#             time.sleep(5)
#             with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 if not exist:
#                     writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)

#         if k == ord('q'):
#             break

# video.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import os
# import csv
# import time
# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from datetime import datetime

# # Initialize video capture and face detector
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Load corrected labels and face data
# with open('data/name.pkl', 'rb') as w:
#     LABELS = pickle.load(w)

# with open('data/face_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# # Convert LABELS to a numpy array
# LABELS = np.array(LABELS)

# # Print shapes of FACES and LABELS to debug the issue
# print("Shape of FACES:", FACES.shape)
# print("Shape of LABELS:", LABELS.shape)

# # Check if the lengths match
# if len(FACES) != len(LABELS):
#     raise ValueError(f"Mismatch in number of samples: FACES has {len(FACES)} samples, LABELS has {len(LABELS)} samples.")

# # Initialize KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# # Get the dimensions of the video frame
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Load the background image and resize it to match the video frame size
# background_path = "img.jpg"  # Change this to the path of your new image
# imagebackground = cv2.imread(background_path)
# imagebackground = cv2.resize(imagebackground, (frame_width, frame_height))

# # Column names for CSV
# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()


# 2nd worked program with label as previously saved

import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier # knn kth nearest neighbour
from datetime import datetime

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load labels and face data
with open('data/name.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Convert LABELS to a numpy array
LABELS = np.array(LABELS)

# Print shapes of FACES and LABELS to debug the issue
print("Shape of FACES:", FACES.shape)
print("Shape of LABELS:", LABELS.shape)

# Check if the lengths match
if len(FACES) != len(LABELS):
    raise ValueError(f"Mismatch in number of samples: FACES has {len(FACES)} samples, LABELS has {len(LABELS)} samples.")

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Column names for CSV
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resize_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 225), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        attendance = [str(output[0]), str(timestamp)]

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == ord('o'):
            time.sleep(5)
            with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

        if k == ord('q'):
            break

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()



