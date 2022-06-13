"""
preprocess_dataset
------------------
Script to apply mtcnn to a dataset for preprocessing
"""
import argparse
import numpy as np
import os
import cv2
from mtcnn.mtcnn import MTCNN

def main(args):
    if not os.path.isdir(args.input):
        raise Exception("Input folder is not a directory")

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Initialize detector and predict boxes
    detector = MTCNN()

    for root, dirs, files in os.walk(args.input, followlinks=args.symbolic_links):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                # Directory name
                label = os.path.join(args.output, os.path.basename(root))

                if not os.path.isdir(label):
                    os.mkdir(label)

                # File name and extension
                file_name, ext = os.path.splitext(file)

                img = cv2.imread(os.path.join(root, file))
                rois = detector.detect_faces(np.array(img))
                for indx, (x, y, w, h) in enumerate([roi["box"] for roi in rois]):
                    face = img[y:y+h, x:x+w]
                    fpath = os.path.join(label, f"{file_name}{indx}{ext}")
                    if not cv2.imwrite(fpath, face):
                        raise Exception(f"Could not write image {fpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to apply mtcnn to a dataset folder to preprocess the images"
    )
    parser.add_argument("-i", "--input", type=str, required=True,
        help="Input folder where all the original images are located.")
    parser.add_argument("-o", "--output", type=str, required=True,
        help="Output folder where all the processed images are going to be saves.")
    parser.add_argument("-s", "--symbolic_links", action="store_true",
        help="If to follow symbolic links in folder (MUST BE VERY CAREFULL OF INF RECURSION!).")
    parser.set_defaults(symbolic_links=False)
    main(parser.parse_args())
