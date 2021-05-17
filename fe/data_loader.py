from __future__ import division
from __future__ import print_function

from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, smart_resize
import os
from PIL import Image as pil_image
import random

_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}


class DirectoryIterator_Modify(DirectoryIterator):

    def __new__(cls, *args, **kwargs):
        return super(DirectoryIterator_Modify, cls).__new__(cls)

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 load_size=(256, 256),
                 crop_mode=None):

        if crop_mode not in {"center", "random", "corner_center_random", None}:
            raise ValueError('Invalid crop mode:', crop_mode,
                             '; expected "center", "random", "corner_center_random" or None.')
        self.crop_mode = crop_mode
        self.load_size = load_size

        super(DirectoryIterator_Modify, self).__init__(
                 directory,
                 image_data_generator,
                 target_size=target_size,
                 color_mode=color_mode,
                 classes=classes,
                 class_mode=class_mode,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 seed=seed,
                 data_format=data_format,
                 save_to_dir=save_to_dir,
                 save_prefix=save_prefix,
                 save_format=save_format,
                 follow_links=follow_links,
                 subset=subset,
                 interpolation=interpolation,
                 dtype=dtype)


    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = self._load_image_modified(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation,
                           load_size=self.load_size,
                           crop_mode = self.crop_mode)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    @staticmethod
    def _load_image_modified(filepath, color_mode,
                           target_size,
                           interpolation,
                           load_size,
                           crop_mode):
        
        if crop_mode is None:
            img =  load_img(filepath,
                            color_mode=color_mode,
                            target_size=target_size,
                            interpolation=interpolation)

        else:
            if isinstance(load_size, int):
                if load_size < target_size[0] or load_size < target_size[1]:
                    raise Exception("Load size must be larger than targe size!")
                img =  load_img(filepath,
                                color_mode=color_mode,
                                target_size=None,
                                interpolation=interpolation)
                raw_width, raw_height = img.size

                if raw_width < raw_height:
                    ratio = load_size / raw_width
                    h = int(raw_height * ratio)
                    img = img.resize(size=(load_size, h), resample=_PIL_INTERPOLATION_METHODS[interpolation])
                else:
                    ratio = load_size / raw_height
                    w = int(raw_width * ratio)
                    img = img.resize(size=(w, load_size), resample=_PIL_INTERPOLATION_METHODS[interpolation])

            elif isinstance(load_size, tuple):
                if load_size[0] < target_size[0] or load_size[1] < target_size[1]:
                    raise Exception("Load size must be larger than targe size!")
                img_size = load_size
                img =  load_img(filepath,
                                color_mode=color_mode,
                                target_size=load_size,
                                interpolation=interpolation)

            else:
                raise Exception("Invalid format of load size!")
            
            if crop_mode == "center":
                img = DirectoryIterator_Modify.crop_img(img, target_size, "center")
            elif crop_mode == "corner_center_random":
                crop_method = random.sample(["center", "left_down_corner", "left_up_corner", "right_down_corner", "right_up_corner"], 1)[0]
                img = DirectoryIterator_Modify.crop_img(img, target_size, crop_method=crop_method)
            elif crop_mode == "random":
                img = DirectoryIterator_Modify.crop_img(img, target_size, crop_method="random")
            else:
                raise Exception()

        return img


    @staticmethod
    def crop_img(img, target_size, crop_method = None):
        if crop_method is None:
            return img

        width, height = img.size
        target_width, target_height = target_size

        if crop_method == "center":
            startx = width//2 - (target_width//2)
            starty = height//2 - (target_height//2)
        
        elif crop_method == "left_down_corner":
            startx = 0
            starty = 0

        elif crop_method == "left_up_corner":
            startx = 0
            starty = height - target_height

        elif crop_method == "right_down_corner":
            startx = width - target_width
            starty = 0

        elif crop_method == "right_up_corner":
            startx = width - target_width
            starty = height - target_height

        else:
            startx = random.randint(0, width - target_width)
            starty = random.randint(0, height - target_height)

        img = img.crop((startx, starty, target_width + startx, target_height + starty))

        return img       

        


class ImageDataGenerator_Modify(ImageDataGenerator):

    def __init__(self,
               featurewise_center=False,
               samplewise_center=False,
               featurewise_std_normalization=False,
               samplewise_std_normalization=False,
               zca_whitening=False,
               zca_epsilon=1e-6,
               rotation_range=0,
               width_shift_range=0.,
               height_shift_range=0.,
               brightness_range=None,
               shear_range=0.,
               zoom_range=0.,
               channel_shift_range=0.,
               fill_mode='nearest',
               cval=0.,
               horizontal_flip=False,
               vertical_flip=False,
               rescale=None,
               preprocessing_function=None,
               data_format=None,
               validation_split=0.0,
               dtype=None):

        super(ImageDataGenerator_Modify, self).__init__(
               featurewise_center = featurewise_center,
               samplewise_center = samplewise_center,
               featurewise_std_normalization = featurewise_std_normalization,
               samplewise_std_normalization = samplewise_std_normalization,
               zca_whitening = zca_whitening,
               zca_epsilon = zca_epsilon,
               rotation_range = rotation_range,
               width_shift_range = width_shift_range,
               height_shift_range = height_shift_range,
               brightness_range = brightness_range,
               shear_range = shear_range,
               zoom_range = zoom_range,
               channel_shift_range=channel_shift_range,
               fill_mode=fill_mode,
               cval=cval,
               horizontal_flip=horizontal_flip,
               vertical_flip=vertical_flip,
               rescale=rescale,
               preprocessing_function=preprocessing_function,
               data_format=data_format,
               validation_split=validation_split,
               dtype=dtype)


    def flow_from_directory(self,
                          directory,
                          target_size=(256, 256),
                          color_mode='rgb',
                          classes=None,
                          class_mode='categorical',
                          batch_size=32,
                          shuffle=True,
                          seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='png',
                          follow_links=False,
                          subset=None,
                          interpolation='nearest',
                          load_size=(256, 256),
                          crop_mode=None):


        return DirectoryIterator_Modify(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            load_size=load_size,
            crop_mode=crop_mode)



if __name__ == "__main__":

    # dicrectory = "/home/workstation/zy/n03954731"
    # dataset_gen = ImageDataGenerator_Modify(samplewise_center = True, samplewise_std_normalization=True, horizontal_flip = True)
    # train_set = dataset_gen.flow_from_directory(dicrectory, target_size = (224, 224), batch_size=50, load_size = 256, crop_mode = "center")
    # for _ in range(2):
    #     print("next", next(train_set)[0].shape)

    from tensorflow.keras.preprocessing.image import smart_resize

