from pathlib import Path
from shutil import rmtree
from subprocess import run
import os
import numpy as np


def monotonize(curves):
    input_shape = curves.shape
    curves = curves.reshape(3, 256)
    for i in range(254, -1, -1):
        curves[:, i] = np.min(curves[:, i:], axis=1)
    return curves.reshape(input_shape)


def get_model(filename):
    "Return a model that outputs a torch tensor with shape [N, 256, C]"

    if Path(filename).suffix == '.pt':
        import torch

        scripted_model = torch.jit.load(filename)

        def model(inputs):
            # channels-first, add batch dim, floatify
            inputs = torch.tensor(inputs.transpose(2, 0, 1)[None, ...], dtype=torch.float32) / 255
            #print("inputs:", inputs.dtype, inputs.min(), inputs.max(), inputs.shape)
            preds = np.array(scripted_model(inputs))
            #print("preds:", preds.shape, preds.dtype)

            # post-process preds
            preds = np.clip(preds, 0, 1)
            preds = monotonize(preds)

            return preds

        for size in (384, 512, 768, 1024, 448, 640):
            input_shape = (1, 3, size, size)
            try:
                _ = scripted_model(torch.zeros(*input_shape, dtype=torch.float32))
                break
            except RuntimeError:
                pass
        model.input_size = input_shape[2:4]

        return model

    elif Path(filename).suffix in ['.keras', '.h5', '.tgz']:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        import tf_keras

        remove_model_folder = False
        if Path(filename).suffix == '.tgz':
            run(f'tar xfz {filename}'.split())
            filename = Path(filename).stem
            assert Path(filename).is_dir(), f'{filename}.tgz does not contain a folder {filename}'
            remove_model_folder = True

        tf_model = tf_keras.models.load_model(filename)
        """TF compatibility issues
            - full h5 is fast, small, and works with all versions (tested: 2.11 ... 2.15), but is "deprecated"
            - keras is superslow and broken with every other version (2.15 fails on 2.13 with ValueError)
            - tf 2.11 does not support keras yet: OSError (assuming HDF5)
            - tfimm requires tf SavedModel format (folder), read as tgz or folder
            - tf 2.14+ takes 4 sec longer for efnv2s_gn load+inference
        """

        if remove_model_folder and Path(filename).is_dir():
            rmtree(filename)

        def model(inputs):
            # add batch dim, floatify
            inputs = tf.constant(inputs[None, ...], dtype=tf.float32) / 255
            preds = tf_model.predict_on_batch(inputs)  # already returns numpy

            # post-process preds
            preds = np.clip(preds, 0, 1)
            preds = monotonize(preds)

            return preds
        model.input_size = getattr(tf_model, 'input_shape', (None, 384, 384, 3))[1:3]

        return model


def get_ensemble(filenames):
    models = [get_model(fn) for fn in filenames]
    input_size = [m.input_size for m in models]
    if all(s == input_size[0] for s in input_size):
        input_size = input_size[0]
    else:
        raise NotImplementedError(f'Models have different input sizes: {", ".join(str(s) for s in input_size)}')

    def ensemble(inputs):
        model_preds = np.stack([model(inputs) for model in models], axis=0)
        preds = np.mean(model_preds, axis=0)

        return preds
    ensemble.input_size = input_size

    return ensemble


def free_curve_map_image(img, curves):
    assert curves.dtype == np.float32, str(curves.dtype)  # float32
    assert img.dtype == np.uint8, f"img.dtype: {img.dtype}"
    assert (curves >= 0).all(), f'curves.min: {curves.min()}'
    assert (curves <= 1).all(), f'curves.max: {curves.max()}'
    curves = curves.reshape(3, 256)

    # Handle gray scale image
    if img.ndim == 2:
        img = img[:, :, None]
        curves = curves.mean(axis=0, keepdims=True)

    transformed = np.empty(img.shape, dtype=curves.dtype)

    # map each channel using fancy indexing
    for i, curve in enumerate(curves):
        transformed[:, :, i] = curve[img[:, :, i]]

    # Remove channel dim from gray scale image
    if transformed.shape[2] == 1:
        transformed = transformed[:, :, 0]

    return transformed * 255  # return as np.float32
