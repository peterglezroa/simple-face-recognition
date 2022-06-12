"""
camera_rep.py
-------------
Script to recognize the face captured by a connected camera
"""
import argparse
import cv2
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import time
from keras_vggface.vggface import VGGFace
from matplotlib import cm
from more_itertools import grouper
from mtcnn.mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.models import Model

# Location of the cascades
CASCADE_PATH = "/home/peterglezroa/Documents/simple-face-recognition/.venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"

# Time between each call of face recognition
REFRESH_TIME = 1
# Max number of faces on which to do the process
MAX_FACES = 5
# Alpha to superimpose image with heatmap
ALPHA=0.4

def face_recognition(shape:list, frame:mp.Array, nfaces:mp.Value, faces:mp.Array,
    heatmap:mp.Array, predictions:mp.Array, model_n:str="vgg16") -> None:
    """
    Function to do face recognition in the detected face frames by the main
    thread.
    This function is intended to be used on separe threads

    It modifies heatmaps to concatenate the resulting heatmaps
    """
    # Initialize detector ----------------------------------------------------
    detector = MTCNN()

    # Initialize recognizer --------------------------------------------------
    model = VGGFace(model=model_n)
    # TODO: Train recognizer with known images

    # Modify model to add heatmap
    conv_output = model.get_layer("conv5_3").output
    pred_output = model.get_layer("fc8/softmax").output
    model = Model(model.input, [conv_output, pred_output])

    # Start loop -------------------------------------------------------
    np_faces = []
    n_faces = 0
    while True:
        # Get frame from shared memory ---------------------------------
        with frame.get_lock():
            np_frame = np.array(frame[:], dtype=np.uint8).reshape(shape)

        # Face detection -----------------------------------------------
        rois = detector.detect_faces(np_frame)
        np_faces.clear()

        # Update shared memory on face location
        with nfaces.get_lock() and faces.get_lock():
            nfaces.value = len(rois[:MAX_FACES])
            # Insert detected faces
            faces[:nfaces.value*4] = np.array(
                [roi["box"] for roi in rois[:MAX_FACES]]).flatten()

#            n_faces = nfaces.value
#            for (x, y, w, h) in grouper(4, faces[:nfaces.value*4],
#            incomplete="ignore"):
#                np_faces.append(
#                    cv2.resize(np_frame[y:y+h, x:x+w].copy(), [224,224],
#                        interpolation = cv2.INTER_AREA)
#                )

        lambs = lambda a, b: a[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
        n_faces = len(rois[:MAX_FACES])
        np_faces = [
            cv2.resize(lambs(np_frame, roi["box"]), [224,224], interpolation=cv2.INTER_AREA) \
            for roi in rois
        ]

        if n_faces > 0:
            # Run predictions with gradient tape
            with tf.GradientTape() as tape:
                last_conv, preds = model(np.array(np_faces))
#                preds_index = np.argmax(pred, axis=1)
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

                # Save permissions to share memory
                with predictions.get_lock():
                    predictions[0] = pred_index

                grads = tape.gradient(class_channel, last_conv)
                pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
                last_conv = last_conv[0]
                conv_heatmap = last_conv @ pooled_grads[..., tf.newaxis]
                conv_heatmap = tf.squeeze(conv_heatmap)
                conv_heatmap = tf.maximum(conv_heatmap, 0)/tf.math.reduce_max(conv_heatmap)
                conv_heatmap = conv_heatmap.numpy()

                # Rescale heatmap to a range 0-255 and recolor
                conv_heatmap = np.uint8(255 * conv_heatmap)
                jet = cm.get_cmap("jet")
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[conv_heatmap]
                jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap = jet_heatmap.resize([faces[2], faces[3]])
                jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    #            superimposed_img = jet_heatmap * ALPHA + np_frame
                np_frame[faces[1]:faces[1]+faces[3], faces[0]:faces[0]+faces[2]] += \
                    (jet_heatmap*ALPHA).astype(np.uint8)
                with heatmap.get_lock():
                    heatmap[:] = np_frame.flatten()

        time.sleep(REFRESH_TIME)

# TODO: namespace
def main(args) -> int:
    # Camera -----------------------------------------------------------------
    # Start camera
    video_cap = cv2.VideoCapture(0)

    # Get a frame to determine dimensions
    ret, frame = video_cap.read()

    # Multiprocessing --------------------------------------------------------
    flattened_frame = frame.flatten()
    freezed_frame = mp.Array("i", flattened_frame) # Image frame that is going to be analyzed
    nfaces = mp.Value("i", 0) # Number of faces detected
    faces = mp.Array("i", MAX_FACES*4) # Coordinates for each face detected
    heatmap = mp.Array("i", len(flattened_frame)) # Resulting heatmap
    predictions = mp.Array("i", MAX_FACES) # Array of predicted classes for each face

    # Start face recognition daemon
    rec_process = mp.Process (
        name="Face Recognition Thread",
        target = face_recognition,
        args = [frame.shape, freezed_frame, nfaces, faces, heatmap, predictions,
            "vgg16"],
        daemon = True
    )
    rec_process.start()

    # Create resulting frame
    res_frame = Image.new("RGB", (frame.shape[1]*2, frame.shape[0]))

    while (True):
        # Capture frame with camera
        ret, frame = video_cap.read()

        # Update shared memory of frame
        with freezed_frame.get_lock():
            freezed_frame[:] = frame.flatten()

        # Draw rectangle for each face and give current prediction
        with nfaces.get_lock() and faces.get_lock() and predictions.get_lock():
            predicted_faces = len(predictions[:])
            for index, (x, y, w, h) in enumerate(grouper(4, faces[:nfaces.value*4])):
                cv2.rectangle(frame, (x, y), (x+w, y+h), [0,255,0], 2)
                # TODO: get label for resulting class
                text = str(predictions[index]) if index < predicted_faces else\
                    "loading..."
                ty = (y-10) if y-10 >= 0 else y
                cv2.putText(frame, text, (x,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0,255,0], 1, cv2.LINE_AA)
#
#        # Append video frame to resulting frame
        res_frame.paste(Image.fromarray(frame), (0,0))
#
        # Append heatmap frame to resulting frame
        with heatmap.get_lock():
            np_heatmap = np.array(heatmap[:], dtype=np.uint8).reshape(frame.shape)
            res_frame.paste(Image.fromarray(np_heatmap), (frame.shape[1], 0))

        cv2.imshow("frame", np.array(res_frame))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Kill dameon
    rec_process.terminate()
    # Close camera
    video_cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to recognize face detected by camera with cv2"
    )
    main(parser.parse_args())
