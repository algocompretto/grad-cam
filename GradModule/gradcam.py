from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        """
        Store the model, the class index is used to measure the class activation map,
        and the layer is used when visualizing it.
        """
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        """
        Finds the final convolutional layer in the network by looping over
        the layers of the network in reverse order.

        :return: Last layer name
        """
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

    def compute_heatmap(self, image, eps=1e-8):
        """
        Construct our gradient model by supplying:
        1. The inputs to our pre-trained model
        2. The output of the final 4D layer in the network
        3. The output of the softmax activations from the model
        """

        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        with tf.GradientTape() as tape:
            """
            Cast the image tensor to a float, pass the image through
            the model and grab the loss associated with specific class
            index.
            """
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

            # Compute the gradients using automatic differentiation
            grads = tape.gradient(loss, convOutputs)

            castConvOutputs = tf.cast(convOutputs > 0, "float32")
            castGrads = tf.cast(grads > 0, "float32")
            guidedGrads = castConvOutputs * castGrads * grads

            # Compute the average of gradient values, use it as weights
            # Compute the ponderation of the filters with respect to the weights
            weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

            # Grab the spatial dimensions of the input and resizes
            # the output class activation map to match dimensions
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(), (w, h))

            # Normalize the heatmap
            num = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = num / denom
            heatmap = (heatmap * 255).astype("uint8")

            return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_RAINBOW):
        """
        Apply the supplied color map to the heatmap and then overlay the heatmap
        with input image.
        """
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)
        return (heatmap, output)
