import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .vgg16 import VGG16
from .birth_point import *
from .feature_entropy import *
import time
import gc
import tensorflow as tf
import random
from .process_dataset import *
from .data_loader import ImageDataGenerator_Modify
import pickle
import tqdm

def load_data(fileURL):
    with open(fileURL, "rb") as handle:
        return pickle.load(handle)

def cumulative_ablation_overall(vggModel, ds, importance_rank):

    # vggModel.evaluate(ds)
    x, y = next(ds)
    layer_outputs = vggModel.intermediate_layer_model.predict(x)
    evaluations = vggModel.predict_model.evaluate(layer_outputs, y, batch_size = 50, verbose = 0)

    unit_mask = np.ones((512,))
    evaluation_list = []
    evaluation_list.append(evaluations)

    for i, item in enumerate(importance_rank):

        unit_mask[importance_rank[i]] = 0
        feature_map = mask_layer_outputs(unit_mask, layer_outputs)

        # print(i)
        preds = vggModel.predict_model.evaluate(feature_map, y, batch_size = 50, verbose = 0)
        evaluation_list.append(preds)        # print(layer_output.shape)

    return evaluation_list


def mask_layer_outputs(unit_mask, layer_outputs):
    unit_mask_tensor = tf.constant(unit_mask, dtype = "float32")
    feature_map = layer_outputs * unit_mask_tensor
    return feature_map


def main():

    ds_path = "/media/workstation/zy/cal_results/imagenet_valset/"
    ds_num = 50 # 50 for valset, 100 for sample set.
    pretrained = True
    save_fname = "ca_100bd_valset.txt"
    save_fname_reverse = "ca_100bd_valset_reverse.txt"

    # vggModel_path = "/media/workstation/zy/model/vgg-dropout2-nol2/weights.22.hdf5"
    vggModel_path = "/home/workstation/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    bp_root_dir = "/media/workstation/zy/cal_results/vgg16/pretrained/"
    save_root_dir = "/media/workstation/zy/cal_results/vgg16/pretrained/"

    vggModel = VGG16()
    vggModel.load_weights(vggModel_path)
    vggModel.compile()
    layer_name = "block5_conv3"
    vggModel.build_intermediate_model(layer_name=layer_name)
    vggModel.build_predict_model()
    vggModel.predict_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc", tf.keras.metrics.top_k_categorical_accuracy])

    for i in range(1000):
        wnid = sorted(os.listdir(ds_path))[i]
        print(wnid)
        # wnid = "n01729322"
        print("Choose %dth wnid "%(i) + wnid + " ...")
        ds = load_directory(os.path.join(ds_path, wnid), sample_number=ds_num, ImageNetLabel=True, VGGPretrainedProcess=pretrained)

        bp = load_data(os.path.join(bp_root_dir, wnid, "bd.pkl"))
        importance_rank = unit_importance_rank(bp)
        # importance_rank.reverse()
        evaluation_list = cumulative_ablation_overall(vggModel, ds, importance_rank)
        np.savetxt(os.path.join(save_root_dir + wnid, save_fname), evaluation_list, fmt="%.6f", delimiter=",")

        # print("Reverse")
        # importance_rank.reverse()
        # evaluation_list = cumulative_ablation_overall(vggModel, ds, importance_rank)
        # np.savetxt(os.path.join(save_root_dir + wnid, save_fname_reverse), evaluation_list, fmt="%.6f", delimiter=",")


    # ds_path = "/media/workstation/zy/cal_results/imagenet_sample/"
    # ds_num = 100 # 50 for valset, 100 for sample set.
    # pretrained = False
    # save_fname = "ca_100bd_100sample.txt"
    # save_fname_reverse = "ca_100bd_100sample_reverse.txt"

    # for i in range(200):
    #     wnid = sorted(os.listdir(ds_path))[i]
    #     print(wnid)
    #     # wnid = "n01729322"
    #     print("Choose %dth wnid "%(i) + wnid + " ...")
    #     ds = load_directory(os.path.join(ds_path, wnid), sample_number=ds_num, ImageNetLabel=True, VGGPretrainedProcess=pretrained)

    #     bp = load_data(os.path.join(bp_root_dir, wnid, "bp_100sample.pkl"))
    #     importance_rank = unit_importance_rank(bp)
    #     # importance_rank.reverse()
    #     evaluation_list = cumulative_ablation_overall(vggModel, ds, importance_rank)
    #     np.savetxt(os.path.join(save_root_dir + wnid, save_fname), evaluation_list, fmt="%.6f", delimiter=",")

    #     print("Reverse")
    #     importance_rank.reverse()
    #     evaluation_list = cumulative_ablation_overall(vggModel, ds, importance_rank)
    #     np.savetxt(os.path.join(save_root_dir + wnid, save_fname_reverse), evaluation_list, fmt="%.6f", delimiter=",")


def one_test():
    ds_path = "/media/workstation/zy/cal_results/imagenet_sample/"
    ds_num = 100 # 50 for valset, 100 for sample set.
    pretrained = False

    vggModel_path = "/media/workstation/zy/model/vgg_dropout1_nol2/weights.18.hdf5"
    bp_root_dir = "/media/workstation/zy/cal_results/vgg16/d-nol2/"

    vggModel = VGG16()
    vggModel.load_weights(vggModel_path)
    vggModel.compile()
    layer_name = "block5_conv3"
    vggModel.build_intermediate_model(layer_name=layer_name)
    vggModel.build_predict_model()
    vggModel.predict_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc", tf.keras.metrics.top_k_categorical_accuracy])
    wnid = sorted(os.listdir(ds_path))[20]
    print(wnid)
    # wnid = "n01729322"
    ds = load_directory(os.path.join(ds_path, wnid), sample_number=ds_num, ImageNetLabel=True, VGGPretrainedProcess=pretrained)

    bp = load_data(os.path.join(bp_root_dir, wnid, "bp_100sample.pkl"))
    importance_rank = unit_importance_rank(bp)
    evaluation_list = cumulative_ablation_overall(vggModel, ds, importance_rank)
    np.savetxt("/home/workstation/zy/github_up/text_temp/fe_20.txt", evaluation_list, fmt="%.6f", delimiter=",")
    # for i in range(2, 20):
        # random.shuffle(importance_rank)
    importance_rank.reverse()
    evaluation_list = cumulative_ablation_overall(vggModel, ds, importance_rank)
    np.savetxt("/home/workstation/zy/github_up/text_temp/fe_20_reverse.txt", evaluation_list, fmt="%.6f", delimiter=",")



if __name__ == "__main__":
    main()
    # one_test()
