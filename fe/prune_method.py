import numpy as np
import random


def L1_norm(Model, layer_name):
    """
        Important list is ranked by the L1 norm of the units.
    """

    layer = Model.get_layer(layer_name)
    k_weights, bias = layer.get_weights()
    l1_norm_arr = np.zeros(shape = (k_weights.shape[-1]))
    for i in range(k_weights.shape[-1]):
        l1_norm_arr[i] = np.sum(np.abs(k_weights[:, :, :, i]))
    importantList = np.argsort(l1_norm_arr)[::-1]
    importantList = importantList.tolist()

    return l1_norm_arr, importantList


def apoz(layer_output, sample_mode = False):
    """
        Important list is ranked by the averaged percentage of zeros of the feature maps.
    """

    apoz_arr = np.zeros(shape = (layer_output.shape[-1]))
    for i in range(layer_output.shape[-1]):
        if sample_mode:
            sample_list = random.sample(range(layer_output.shape[0]), 2)
            layer_output_slice = layer_output[sample_list, :, :, i]
        else:
            layer_output_slice = layer_output[:, :, :, i]

        zero_count = np.count_nonzero(layer_output_slice == 0)
        apoz_arr[i] = zero_count / (layer_output.shape[-3] * layer_output.shape[-2] * layer_output.shape[-1])

    importantList = np.argsort(apoz_arr)[::-1]
    importantList = importantList.tolist()
    importantList.reverse()

    return apoz_arr, importantList


def average_mean_method(layer_output, sample_mode = False):
    """
        Important list is ranked by the averaged mean of the feature maps.
    """

    mean_arr = np.zeros(shape = (layer_output.shape[-1]))
    for i in range(layer_output.shape[-1]):
        if sample_mode:
            sample_list = random.sample(range(layer_output.shape[0]), 2)
            layer_output_slice = layer_output[sample_list, :, :, i]
        else:
            layer_output_slice = layer_output[:, :, :, i]

        mean_value = np.mean(layer_output_slice)
        mean_arr[i] = mean_value

    importantList = np.argsort(mean_arr)[::-1]
    importantList = importantList.tolist()

    return mean_arr, importantList

