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

        pass

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_RAINBOW):
        """
        Apply the supplied color map to the heatmap and then overlay the heatmap
        with input image.
        """
        pass

