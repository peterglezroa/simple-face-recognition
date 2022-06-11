"""
camera_rep.py
-------------
Script to recognize the face captured by a connected camera
"""
import argparse
import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import Model

def main():
    # Initialize detector
    detector = MTCNN()

    # Initialize recognizer
    model = VGGFace(model="VGG16")
    # TODO: Train recognizer with known images
    # TODO: Freezear todas las capas expecto el softmax
    # TODO: Entrenar

    # Modify model to add heatmap
    print(model.summary())
    conv_output = model.get_layer("fc7").output
    pred_output = model.get_layer("fc8/softmax").output
    model = Model(model.input, output=[conv_output, pred_output])

    # Use cv2 camera
    video_cap = cv2.VideCapture(0)

    while (True):
        # Capture frame by frame from camera
        ret, frame = video_cap.read()

        # Detect faces in the frame
        # TODO: Use a thread
        rois = detector.detect_faces(frame)

        # Extract one face
        # TODO: Do for every face
        x, y, w, h = rois[0]["box"]
        roi = image[y:y+h, x:x+h]
        roi = cv2.resize(roi, (224, 224), interpolation = cv2.INTER_AREA)
        roi = np.expand_dims(roi, axis=0).astype(np.float32)

        # Predict
        conv, pred = model.predict(roi)
        target = np.argmax(pred, axis=1).squeeze()
        w, b = model.get_layer("fc8/softmax").weights
        weights = w[:, target].numpy()
        heatmap = conv.squeeze() @ weights
        cv2.imshow("face", zoom(conv, zoom=[224,224]))

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0,255,0], 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Close camera
    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to recognize face detected by camera with cv2"
    )
