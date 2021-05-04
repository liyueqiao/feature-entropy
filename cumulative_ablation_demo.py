import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from vgg16 import VGG16
from birth_point import *
from feature_entropy import *
import time
import gc
import tensorflow as tf
import random
from process_dataset import *



def cumulative_ablation(Model,
                        layer_outputs,
                        dataset,
                        workDirectory,
                        baseDirectory,
                        batch_size = 100,
                        Reverse = False,
                        UseParallel = True,
                        layer_name = "block5_conv3"):

    """
        Cumulative ablation test.

        Parameters
        ----------
        Model: Tensorflow model.
            The test model. The test involves an operation of assigning the weights of units to all zeros. 
            In this function, we write the ablation operation as a class method in VGG16 class, which is Model.change_intermediate_weights method.
        
        layer_outputs: 4-D array. [img_idx, channel_idx, x, y]
            The output of a given layer. The data format is "channel_last".

        dataset: tuple. (x, y). x: 4-D array [img_idx, channel_idx, i, j]; y: categorial label.
            The dataset need to be evaluated during cumulative ablation. The data format is "channel_last".

        workDirectory: path-like string.
            Path to the directory that generates the intermediate files.

        baseDirectory: path-like string.
            Path to the code directory.
        
        batch_size: int.
            The batch size of evaluation during cumulative ablation.

        Reverse: bool.
            Flag to conduct cumulative ablation or its reverse test. The default value is False.

        UseParallel: bool.
            Flag to use parallel in birth point calculations. The default value is True.

        layer_name: str.
            The name of the layer to be ablated. 

        Returns
        -------
        evaluation_list: list.
            The list record the performace change during cumulative abalation. Each element consists of loss, accuracy and top-5 accuracy.

    """

    x, y = dataset
    _, bp = calculate_birth_point(layer_outputs, workDirectory=workDirectory, baseDirectory=baseDirectory, UseParallel=UseParallel)
    importance_rank = unit_importance_rank(bp)
    if Reverse:
        print("Reverse the importance list...") 
        importance_rank.reverse()
    evaluation_list = []
    print("Evaluate on unchanged model...")
    evaluations = Model.evaluate(x, y, batch_size = batch_size, verbose = 1)
    evaluation_list.append(evaluations)

    for i, item in enumerate(importance_rank):
        print("Ablating the %dth unit: Unit %d"%(i, item))
        Model.change_intermediate_weights([item], layer_name)
        evaluations = Model.evaluate(x, y, batch_size = batch_size,  verbose = 1)
        evaluation_list.append(evaluations)

    del Model
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return evaluation_list


def demo():

    # Initialize dataset.
    ds_path = "path-to-your-image-parent-folder"
    ds_path = "/home/workthu/zy/cal_results/imagenet_sample/"
    wnid = random.sample(os.listdir(ds_path), 1)[0]
    print("Choose wnid " + wnid + " ...")
    ds_gen = load_directory(os.path.join(ds_path, wnid), sample_number=100, ImageNetLabel=True, VGGPretrainedProcess=True)
    dataset = next(ds_gen)

    # Initialize direcotry arguments
    current_time = time.strftime("%H-%M-%S")
    baseDirectory = os.path.dirname(os.path.realpath(__file__))
    workDirectory = os.path.join(baseDirectory, current_time + "-txttrash")
    try:
        os.mkdir(workDirectory)
    except:
        raise Exception("Directory already exists!")

    # Initialize model and solve intermediate outputs
    # vggModel_path = "~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    vggModel_path = "/home/workthu/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    vggModel = VGG16()
    vggModel.load_weights(vggModel_path)
    vggModel.compile()
    layer_name = "block5_conv3"
    vggModel.build_intermediate_model(layer_name=layer_name)
    layer_outputs = vggModel.get_intermediate_layer_output(dataset)

    # Cumulative ablation test
    evaluation_list = cumulative_ablation(vggModel, layer_outputs, dataset, workDirectory, baseDirectory, batch_size= 100,
                    Reverse=True, UseParallel=True, layer_name=layer_name)

    # Save results
    np.savetxt(os.path.join(baseDirectory, "results.csv"), evaluation_list, fmt="%.6f", delimiter=",")

    # Complete 
    os.rmdir(workDirectory)
    print("Task complete!")

if __name__ == "__main__":
    demo()

