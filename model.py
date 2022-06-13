"""
model.py
--------
Script and functions to create a model for face recognition of a defined amount of people.
"""
import argparse
import os
from datetime import datetime
from keras_vggface.vggface import VGGFace
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(base_model_name:str, num_people:int) -> Model:
    # Load base model
    base_model = VGGFace(model=base_model_name)
    if base_model is None:
        raise Exception(f"Not a valid model name for VGGFace {base_model}")

    # Remove model softmax
    base_model = Model(inputs=base_model.input,
        outputs=base_model.layers[-2].output, name=base_model_name)
    base_model.trainable = False

    # Freeze base model
    model = Sequential(name=f"modified_{base_model_name}")
#    model.add(InputLayer(input_shape=[224,224,3]))
    model.add(base_model)
    model.add(Dense(num_people, activation="softmax", name="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy",
        metrics=[CategoricalAccuracy()])
    print(model.summary())
    return model

def train_model(model:Model, dataset:str, epochs:int, path:str) -> Model:
    # Create Image augmentation
    datagen = ImageDataGenerator (
        rotation_range=20,
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

    # Get dataset
    train_ds = datagen.flow_from_directory (
        directory = dataset,
        target_size = [224, 224],
        class_mode = "categorical",
        batch_size = 32,
    )

    model.fit(train_ds, epochs=epochs)
    model.save(path)
    return model

def main(args):
    if os.path.splitext(args.file)[1] != ".h5":
        raise Exception("This script only accepts .h5 files for models")

    # Create or load model
    model = create_model(args.create, args.num_people) if (args.create is not None and
        args.num_people > 0) else load_model(args.file)

    # Train model
    train_model(model, args.dataset, args.epochs, args.file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create and train a model with keras"
    )
    parser.add_argument("-c", "--create", type=str, required=False,
        help="Base model on which a model will be create.")
    parser.add_argument("-n", "--num_people", type=int, default=0,
        help="The amount of people for the output of the new model.")
    parser.add_argument("-f", "--file", type=str, required=True,
        help="File path where the model will be saved or loaded.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
        help="The dataset on which the model will be trained.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
        help="The amount of epochs to train the model")
    main(parser.parse_args())
