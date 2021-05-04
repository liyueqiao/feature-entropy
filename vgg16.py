import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


class VGG16(keras.models.Sequential):

    def __init__(self,
                input_shape = (224, 224, 3)):
        super().__init__()
        self.build(input_shape)

    def build_intermediate_model(self, layer_name):
        self.intermediate_layer_model = keras.models.Model(inputs=self.input, outputs=self.get_layer(layer_name).output)

    def build_predict_model(self, input_shape = None):
        if input_shape is None: input_shape = (14, 14, 512)
        inputs = tf.keras.layers.Input(input_shape)
        x = self.get_layer("block5_pool")(inputs)
        x = self.get_layer("flatten")(x)
        x = self.get_layer("fc1")(x)
        x = self.get_layer("fc2")(x)
        x = self.get_layer("predictions")(x)

        self.predict_model =  keras.models.Model(inputs, x, name="predict_model")


    def build(self, input_shape):
        self.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        #Block 2
        self.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        self.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        
        #Block 3
        self.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        self.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        self.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        #Block 4
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        
        #Block 5
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        #Fully connected
        self.add(Flatten(name="flatten"))
        # self.add(Dropout(0.3))
        self.add(Dense(4096,activation= "relu",  name='fc1'))
        self.add(keras.layers.Activation("relu"))

        self.add(Dropout(0.5))
        self.add(Dense(4096, activation= "relu", name='fc2'))
        # self.add(keras.layers.Activation("relu"))
        self.add(Dropout(0.5))

        self.add(Dense(1000, activation="softmax", name='predictions'))


    def get_intermediate_layer_output(self, ds):
        img_num = ds[0].shape[0]

        # avoid of OOM
        if img_num > 100:
            i = 0
            while True:
                sub_ds = (ds[0][100 * i : 100 * i + 100, :, :, :], ds[1][100 * i : 100 * i + 100, :])
                layer_outputs_temp = self.output_layers(self.intermediate_layer_model, sub_ds)
                if i == 0:
                    layer_outputs = layer_outputs_temp
                else:
                    layer_outputs = np.vstack((layer_outputs, layer_outputs_temp))  
                i += 1
                if i >= img_num//100:
                    sub_ds = (ds[0][100 * i :, :, :, :], ds[1][100 * i :, :])
                    layer_outputs_temp = self.output_layers(self.intermediate_layer_model, sub_ds)
                    layer_outputs = np.vstack((layer_outputs, layer_outputs_temp))
                    return layer_outputs

        else:
            layer_outputs = self.output_layers(self.intermediate_layer_model, ds)
            return layer_outputs.numpy()

    @tf.function(experimental_relax_shapes=True)
    def output_layers(self, model, x):
        layer_outs = model(x)
        return layer_outs

    def change_intermediate_weights(self, filterList = [], layer_name = "block5_conv3"):
        assert isinstance(filterList, list)
        # weights = np.array(self.get_weights())
        weights = self.get_layer(name = layer_name).get_weights()
        zero_filter = np.zeros((3, 3, 1))
        for filter_idx in filterList:
            weights[0][:, :, :, filter_idx] = zero_filter
            weights[1][filter_idx] = 0
        self.get_layer(name = layer_name).set_weights(weights)

    def compile(self):
        super().compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc", keras.metrics.top_k_categorical_accuracy])


if __name__ == "__main__":
    from process_dataset import *
    from data_loader import ImageDataGenerator_Modify

    vggModel_path = "/home/workthu/zy/code/vgg_test/model/vgg_imagenet_train/vgg-dropout1-nol2/weights.18.hdf5"
    # vggModel_path = "/home/workthu/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    ds_path = "/home/workthu/zy/cal_results/imagenet_sample/"

    vggModel = VGG16()
    vggModel.load_weights(vggModel_path)
    vggModel.compile()
    ds = load_directory(ds_path, sample_number=100, 
                ImageNetLabel=True, VGGPretrainedProcess=False)

    vggModel.build_intermediate_model("block5_conv3")
    outputs = vggModel.get_intermediate_layer_output(next(ds))
    print(outputs.shape)
    # vggModel.evaluate(ds, steps = 200)
