import pickle
import re
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


class MyImage:
    def __init__(self):
        self.image_name = ""

    def clear(self):
        self.image_name = ""

    def set_image_path(self):
        root = Tk()
        path = filedialog.askopenfilename()
        self.image_name = path
        root.destroy()
        if self.image_name == "":
            return False
        return True

    def get_image_path(self):
        return self.image_name

    def load_model_and_classify(self, args, model_name, labelbin_name):
        model = load_model(args[model_name])
        lb = pickle.load(open(args[labelbin_name], "rb"))

        image = cv2.imread(args["image"])
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predict = model.predict(image)[0]
        idx = np.argmax(predict)
        label = lb.classes_[idx]

        return label, predict[idx]

    def process_label(self, label):
        if 'Banana' in label.split():
            return 'Banana'
        try:
            int(label.split(" ")[-1])
            label = " ".join(label.split(" ")[:-1])
        except Exception as e:
            pass
        return label

    def checkImage(self):
        root = Tk()
        if self.image_name == "":
            messagebox.showinfo("Action window", "You need to select image path first")
            root.destroy()
            return

        # data path:
        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--model", default='F:/Lior Saghi/Downloads/archive11/my_model6',
                        help="path to trained model model")
        ap.add_argument("-l", "--labelbin", default='F:/Lior Saghi/Downloads/archive11/lb6.pickle',
                        help="path to label binarizer")
        ap.add_argument("-i", "--image", default=self.get_image_path(),
                        help="path to input image")
        args = vars(ap.parse_args())

        ap2 = argparse.ArgumentParser()
        ap2.add_argument("-m", "--model2", default='F:/Lior Saghi/Downloads/archive11/my_model_banana3',
                         help="path to trained model model")
        ap2.add_argument("-l", "--labelbin2", default='F:/Lior Saghi/Downloads/archive11/lb_banana3.pickle',
                         help="path to label binarizer")
        ap2.add_argument("-i", "--image", default=self.get_image_path(),
                         help="path to input image")
        args2 = vars(ap2.parse_args())

        # Model loading and image classifying
        print("Loading network and classifying the image-")
        label, predict = self.load_model_and_classify(args, "model", "labelbin")
        label2, predict2 = self.load_model_and_classify(args2, "model2", "labelbin2")

        # Label processing
        label = self.process_label(label)
        label2 = self.process_label(label2)

        # Load the original image
        output = cv2.imread(args["image"]).copy()
        output = imutils.resize(output, width=400)
        label_org = (5, output.shape[0] - 10)
        label_banana_org = label_org
        labels_color = (178, 90, 38)

        # Get the label1 size
        (text_width1, text_height1), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width2 = text_width1

        if "Banana" in label:
            label3 = self.banana_info(label2)
            label_org = (5, output.shape[0] - 40)
            label_banana_org = (5, output.shape[0] - 10)
            # Get the label3 size
            (text_width2, text_height2), baseline = cv2.getTextSize(label3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the coordinates for the background rectangle
        padding = 5
        background_color = (255, 255, 255)
        background_org = (label_org[0] - padding, label_org[1] - text_height1 - padding)
        background_end = (label_org[0] + text_width2 + padding, label_banana_org[1] + padding)

        # Draw the background rectangle
        cv2.rectangle(output, background_org, background_end, background_color, cv2.FILLED)

        # Fruit type label
        cv2.putText(output, label, label_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, labels_color, 2)
        print("Fruit type: {}".format(label))

        # Draw labels
        if "Banana" in label:
            cv2.putText(output, label3, (label_banana_org[0], label_banana_org[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        labels_color, 1)
            print("Banana day: {} \n".format(label2))

        cv2.imshow("Output", output)
        cv2.waitKey(0)
        root.destroy()

    def banana_info(self, label):
        label2 = label.split("/")[-1]
        matches = re.findall(r'\d+', label2)
        number = int(matches[0])
        if 9 - number == 0:
            return "Eat the banana Today!"
        elif 9 - number < 0:
            return "It is not recommended to eat the banana"
        elif 9 - number > 7:
            return f"Wait {3 - number} days to eat the banana"
        else:
            return f"Approx. {9 - number} more days to eat the banana"
