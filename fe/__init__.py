from .compute_betti_curve import BC
from .birth_point import calculate_birth_point
from .feature_entropy import calculate_feature_entropy, unit_importance_rank, calculate_feature_entropy_for_array
from .process_dataset import load_directory
from .prune_method import L1_norm, apoz, average_mean_method

MODEL_LIST = {
    "pretrained": "/home/workstation/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
    "vgg_dropout1_nol2": "/media/workstation/zy/model/new_vgg_2_22/",
    "vgg_momentum_1_7": "/media/workstation/zy/model/vgg_momentum_1_7/",
    "vgg_add_val_model": "/media/workstation/zy/model/vgg_add_val_model/",
    "d-nl2":"/media/workstation/zy/model/vgg_dropout1_nol2/",
    "nd-nl2":"/media/workstation/zy/model/vgg-nodropout-nol2/",
    "nd-nl2-15":"/media/workstation/zy/model/vgg-nodropout-nol2/",
    "nd-l2":"/media/workstation/zy/model/vgg-nodropout-l2norm/",
    "nd-l2-15":"/media/workstation/zy/model/vgg-nodropout-l2norm/",
    "vgg_bn_1_12":"/media/workstation/zy/model/vgg_bn_1_12/",
}

import pickle

def load_data(fileURL):
    with open(fileURL, "rb") as handle:
        return pickle.load(handle)

def save_data(fileURL, data):
    with open(fileURL, "wb") as handle:
        pickle.dump(data, handle)