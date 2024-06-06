
# import numpy as np
# import os
# import pickle
# import cv2

# # Open the video capture
# video = cv2.VideoCapture(0)

# # Load the Haar cascade classifier
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Initialize an empty list to store face data
# face_data = []

# # Prompt the user for their name
# name = input("Enter Your Name: ")

# while True:
#     # Read a frame from the video capture
#     ret, frame = video.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     # Loop through the detected faces
#     for (x, y, w, h) in faces:
#         # Crop the detected face from the frame
#         crop_img = frame[y:y + h, x:x + w]

#         # Resize the cropped face to 50x50 pixels
#         resize_img = cv2.resize(crop_img, (50, 50))

#         # Append the resized face image to the face_data list if less than 100 samples are collected
#         if len(face_data) < 100:
#             face_data.append(resize_img)
#             # Draw a rectangle around the detected face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
#             # Display the number of collected samples on the frame
#             cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2)
#         else:
#             break

#     # Display the frame with the detected face and count
#     cv2.imshow("frame", frame)

#     # Break the loop if 'q' is pressed or 100 samples are collected
#     if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= 100:
#         break

# # Release the video capture and close all OpenCV windows
# video.release()
# cv2.destroyAllWindows()



# # save the faces in pickle file
# face_data = np.array(face_data)
# face_data = face_data.reshape(100, -1)

# if 'names.pkl' not in os.listdir('data/'):
#     names = [name]* 100
#     with open('data/name.pkl', 'wb') as f:
#         pickle.dump(names,f)
# else:
#     with open('data/names.pkl', 'rb') as f:
#         names = pickle.load(f)
#         names = names + [name]*100
#     with open ('data/names.pkl', 'wb') as f:
#         pickle.dump(names.f)
# if 'face_data.pkl' not in os.listdir('data/'):
#     with open('data/face_data.pkl', 'wb') as f:
#         pickle.dump(face_data, f)
# else:
#     with open('data/face_data.pkl', 'rb') as f:
#         faces = pickle.load(f)
#     faces = np.append(faces, face_data, axis = 0)
#     with open('data/face_data.pkl', 'wb') as f:
#         pickle.dump(faces,f)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# import numpy as np
# import os
# import pickle
# import cv2

# # Create the data directory if it doesn't exist
# if not os.path.exists('data'):
#     os.makedirs('data')

# # Open the video capture
# video = cv2.VideoCapture(0)

# # Load the Haar cascade classifier
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Initialize an empty list to store face data
# face_data = []

# # Prompt the user for their name
# name = input("Enter Your Name: ")

# while True:
#     # Read a frame from the video capture
#     ret, frame = video.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     # Loop through the detected faces
#     for (x, y, w, h) in faces:
#         # Crop the detected face from the frame
#         crop_img = frame[y:y + h, x:x + w]

#         # Resize the cropped face to 50x50 pixels
#         resize_img = cv2.resize(crop_img, (50, 50))

#         # Append the resized face image to the face_data list if less than 100 samples are collected
#         if len(face_data) < 100:
#             face_data.append(resize_img)
#             # Draw a rectangle around the detected face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
#             # Display the number of collected samples on the frame
#             cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2)
#         else:
#             break

#     # Display the frame with the detected face and count
#     cv2.imshow("frame", frame)

#     # Break the loop if 'q' is pressed or 100 samples are collected
#     if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= 100:
#         break

# # Release the video capture and close all OpenCV windows
# video.release()
# cv2.destroyAllWindows()

# # Convert the face_data list to a numpy array and reshape
# face_data = np.array(face_data)
# face_data = face_data.reshape(100, -1)

# # Load existing labels and face data if they exist, otherwise initialize them
# if os.path.exists('data/name.pkl'):
#     with open('data/name.pkl', 'rb') as f:
#         names = pickle.load(f)
# else:
#     names = []

# names.extend([name] * 100)

# if os.path.exists('data/face_data.pkl'):
#     with open('data/face_data.pkl', 'rb') as f:
#         faces = pickle.load(f)
#     faces = np.vstack((faces, face_data))
# else:
#     faces = face_data

# # Save the updated labels and face data
# with open('data/name.pkl', 'wb') as f:
#     pickle.dump(names, f)

# with open('data/face_data.pkl', 'wb') as f:
#     pickle.dump(faces, f)

# print("Face data and labels saved successfully.")



# Traing 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222



import numpy as np
import os
import pickle
import cv2

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Open the video capture
video = cv2.VideoCapture(0)

# Load the Haar cascade classifier
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize an empty list to store face data
face_data = []

# Prompt the user for their name
name = input("Enter Your Name: ")

while True:
    # Read a frame from the video capture
    ret, frame = video.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Crop the detected face from the frame
        crop_img = frame[y:y + h, x:x + w]

        # Resize the cropped face to 50x50 pixels
        resize_img = cv2.resize(crop_img, (50, 50))

        # Append the resized face image to the face_data list if less than 100 samples are collected
        if len(face_data) < 100:
            face_data.append(resize_img)
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            # Display the number of collected samples on the frame
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2)
        else:
            break

    # Display the frame with the detected face and count
    cv2.imshow("frame", frame)

    # Break the loop if 'q' is pressed or 100 samples are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= 100:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert the face_data list to a numpy array and reshape
face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)

# Load existing labels and face data if they exist, otherwise initialize them
if os.path.exists('data/name.pkl'):
    with open('data/name.pkl', 'rb') as f:
        names = pickle.load(f)
else:
    names = []

names.extend([name] * 100)

if os.path.exists('data/face_data.pkl'):
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.vstack((faces, face_data))
else:
    faces = face_data

# Print shapes to debug
print(f"Shape of collected face data: {face_data.shape}")
print(f"Total faces collected: {faces.shape}")
print(f"Total names collected: {len(names)}")

# Save the updated labels and face data
with open('data/name.pkl', 'wb') as f:
    pickle.dump(names, f)

with open('data/face_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

print("Face data and labels saved successfully.")


