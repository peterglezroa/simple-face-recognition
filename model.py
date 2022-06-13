import argparse
import os
from datetime import datetime
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.model import Model, load_model
from tensorflow.keras.utils import image_dataset_from_directory

def create_model(base_model_name:str) -> Model:
    # Load base model
    base_model = VGGFace(model=base_model_name, include_top=False)
    if base_model is None:
        raise Exception(f"Not a valid model name for VGGFace {base_model}")

    # Freeze base model
    inputs = Input(shape=(224, 224, 3))
    model = base_model(inputs, training=False)

    # Add new softmax and compile new model
    outputs = Dense(3, activation="softmax")(model)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
        metrics=[CategoricalAccuracy()])
    print(model.summary())
    return model

def train_model(model:Model, dataset:str, epochs:int, path:str) -> Model:
    train_ds = image_dateset_from_directory (
        directory = dataset,
        labels = "inferred",
        label_mode = "categorical",
        batch_size = 32,
        image_size = [224, 224],
    )

    model.fit(train_ds, epochs=epochs)
    model.save(path)
    return model

def main(args):
    if os.path.splitext(args.file)[1] != ".h5":
        raise Exception("This script only accepts .h5 files for models")

    # Create or load model
    model = create_model(args.create, args.file) if args.create is not None else \
        load_model(args.file)

    # Train model
    train_model(model, args.dataset, args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create and train a model with keras"
    )
    parser.add_argument("-c", "--create", type=str, required=False,
        help="Base model on which a model will be create.")
    parser.add_argument("-f", "--file", type=str, required=True,
        help="File path where the model will be saved or loaded.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
        help="The dataset on which the model will be trained.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
        help="The amount of epochs to train the model")
    main(parser.parse_args())
