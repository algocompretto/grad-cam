from GradModule.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50, VGG16, imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg",
                choices=("vgg", "resnet"),
                help="model to be used")
args = vars(ap.parse_args())

Model = VGG16

if args["model"] == "resnet":
    Model = ResNet50

# Load pre-trained CNN
print("[INFO] Loading pretrained model...")
model = Model(weights="imagenet")

print("[INFO] Loading and preprocessing the image...")
# Load the original image and resizes
orig = cv2.imread(args["image"])
resized = cv2.resize(orig, (224, 224))

# Load the image from disk and preprocess
image = load_img(args["image"], target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

print("[INFO] Predicting...")
# Networks predictions
preds = model.predict(image)
i = np.argmax(preds[0])

print("[INFO] Decoding predictions...")
# Decode predictions
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = f"{label}: {prob * 100:.2f}%"
print(f"[INFO] {label}")

# Initialize GradCAM
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)

heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.65)

# Draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2)

# Display the original image and resulting heatmap and output image
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)
