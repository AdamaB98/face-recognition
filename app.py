import streamlit as st
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder


# Specify the paths to your models and image directory
PROTO_PATH = 'deploy.prototxt.txt'
MODEL_PATH = 'res10_300x300_ssd_iter_140000.caffemodel'
EMBEDDER_PATH = 'openface.nn4.small2.v1.t7'
SERIALIZED_DATA_PATH = 'serialized_data.pkl'
RECOGNIZER_PATH = './model/recognizer.pkl'
LABEL_ENCODE_PATH = 'labels.pkl'
IMAGE_DIRECTORY = 'images'

# Load the face detector
detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

data = pickle.loads(open(SERIALIZED_DATA_PATH, "rb").read())

le = LabelEncoder()
le.fit_transform(data["names"])

# Load the face embedder
embedder = cv2.dnn.readNetFromTorch(EMBEDDER_PATH)


# Load the recognizer and label encoder for face recognition
recognizer = pickle.loads(open(RECOGNIZER_PATH, 'rb').read())

def recognize(detections):
    print("Recognizing")
    for i in range(0, detections.shape[2]):
	    # extract the confidence (i.e., probability) associated with the
	    # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output image
    print("Showing Image")
    st.image(image, caption="Detection", use_column_width=True)

# Create a web application using Streamlit
st.title('Face Recognition App')
st.write('Upload an image below to recognize faces')

# Create a file uploader for the image
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        recognize(detections)
    else:
        st.error("Error: Failed to load the image.")
