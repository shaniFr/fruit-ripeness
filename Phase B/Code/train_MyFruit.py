import argparse
import cv2
import matplotlib
import matplotlib
import numpy as np
import os
import pickle
import random
from keras.metrics import Recall, Precision
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from vggnet16 import VGGNet16
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imutils import paths

matplotlib.use('TkAgg')


def data_path():
    ap = argparse.ArgumentParser()
    model_dir_path = 'F:/Lior Saghi/Downloads/archive11/my_model_banana3'
    ap.add_argument("-d", "--dataset",
                    default='F:/Lior Saghi/Downloads/New_bananas_image_Shani/New_bananas_image_Shani/',
                    required=False,
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", default=model_dir_path, required=False,
                    help="path to output model")
    ap.add_argument("-l", "--labelbin", default='F:/Lior Saghi/Downloads/archive11/lb_banana3.pickle',
                    required=False,
                    help="path to output label binarizer")
    return vars(ap.parse_args())


def load_images(image_paths):
    # initialize the data and labels
    data, labels = [], []
    for imagePath in image_paths:  # loop over each image path
        image = cv2.imread(imagePath)  # load image from the path.
        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))  # resize the image
        image = img_to_array(image)  # convert the image to array format using Keras utility
        data.append(image)  # append the image array to data list
        # extract the label of the image and append to labels list
        label = os.path.basename(os.path.dirname(imagePath))
        labels.append(label)
        # convert data and labels lists to numpy arrays, scale pixel intensities to the range [0, 1]
    return np.array(data, dtype="float") / 255.0, np.array(labels)


def train_model(trainX, trainY, testX, testY):
    # initialize image data generator
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    # build a VGGNet16 model with given image dimensions
    model = VGGNet16.build_model(width=IMAGE_SIZE[1], height=IMAGE_SIZE[0],
                                 depth=IMAGE_SIZE[2], classes=len(lb.classes_))
    # initialize an Adam optimizer with learning rate decay
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    # compile the model with categorical crossentropy loss, Adam optimizer and metrics
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy", Recall(), Precision()])
    # fit the model using data augmentation generator, with validation data and verbose output
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)
    return model, H


def plot_results(history):
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.show()

    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.show()

    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # summarize history for recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # summarize history for precision
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    prediction = model.predict(testX)
    matrix = confusion_matrix(testY.argmax(axis=1), prediction.argmax(axis=1))
    class_names = [c.split(os.path.sep)[-1] for c in lb.classes_]  # extract class names from paths

    df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names)  # use class names here

    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  # Adjust layout to make sure labels are not cut off
    plt.show()

    cv2.waitKey(0)


if __name__ == "__main__":
    # initialize the number of epochs to train for, initial learning rate,
    # batch size, and image dimensions
    EPOCHS = 50
    INIT_LR = 1e-3
    BS = 32
    IMAGE_SIZE = (96, 96, 3)
    args = data_path()
    # Check if dataset path exists
    if not os.path.exists(args["dataset"]):
        print(f"Dataset path {args['dataset']} does not exist.")
        exit()

    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    if not imagePaths:
        print(f"No images found in the provided path {args['dataset']}.")
        exit()
    random.seed(42)
    random.shuffle(imagePaths)

    data, labels = load_images(imagePaths)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # Debugging - print out labels and lb.classes_
    print(labels)
    print(lb.classes_)

    # Divide the data into training and testing splits using 80% of the data for training and the remaining 20% for
    # testing
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.2, random_state=42)

    model, H = train_model(trainX, trainY, testX, testY)

    # save the model to disk
    model.save(args["model"])

    # save the label binarizer to disk
    with open(args["labelbin"], "wb") as f:
        f.write(pickle.dumps(lb))

    plot_results(H)
