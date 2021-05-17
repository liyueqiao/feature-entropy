import os
import random
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from .data_loader import ImageDataGenerator_Modify


def load_directory(ds_path, sample_number = 100, ImageNetLabel = False, VGGPretrainedProcess = True, load_size = 256, crop_mode = None):
    if not VGGPretrainedProcess:
        dg = ImageDataGenerator_Modify(samplewise_center = True, samplewise_std_normalization=True,)
    else:
        dg = ImageDataGenerator_Modify(preprocessing_function=preprocess_input)
    if not ImageNetLabel:
        ds_gen = dg.flow_from_directory(ds_path, target_size = (224, 224), shuffle = False, batch_size=sample_number, class_mode="categorical", load_size=load_size, crop_mode=crop_mode)
        return ds_gen
    else:
        ds_gen = dg.flow_from_directory(ds_path, target_size = (224, 224), shuffle = False, batch_size=sample_number, class_mode="sparse", load_size=load_size, crop_mode=crop_mode)
        kerasLabelList = lookup_keraslabel(ds_path)
        ds_gen = yield_imagenet_generator(ds_gen, kerasLabelList)
        return ds_gen


def lookup_keraslabel(ds_path):
    map_clsloc_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "map_clsloc.txt")
    with open(map_clsloc_fname, "r") as f:
        map_clsloc_read = f.readlines()
    word_id_list = sorted([item.split()[0] for item in map_clsloc_read])
    label_list = sorted(os.listdir(ds_path))
    kerasLabel = [word_id_list.index(item) for item in label_list]
    return kerasLabel


def yield_imagenet_generator(ds_gen, kerasLabelList):
    for x, y in ds_gen:
        y_change = np.asarray([kerasLabelList[int(i)] for i in y])
        y_batch = to_categorical(y_change, 1000)
        yield (x, y_batch)


if __name__ == "__main__":
    pass