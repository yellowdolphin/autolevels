from pathlib import Path
from shutil import rmtree
from subprocess import run
import os
import numpy as np
import h5py


def monotonize(curves):
    input_shape = curves.shape
    curves = curves.reshape(3, 256)
    for i in range(254, -1, -1):
        curves[:, i] = np.min(curves[:, i:], axis=1)
    return curves.reshape(input_shape)


def get_model(filename):
    """Return a model that outputs a torch tensor with shape [N, 256, C]"""

    if Path(filename).suffix == '.pt':
        import torch

        scripted_model = torch.jit.load(filename)

        def model(inputs):
            # channels-first, add batch dim, floatify
            assert inputs.dtype in {np.dtype('uint8'), np.dtype('uint16')}, f'input type {inputs.dtype} not supported'
            maxvalue = 65535 if inputs.dtype == np.uint16 else 255
            inputs = torch.tensor(inputs.transpose(2, 0, 1)[None, ...], dtype=torch.float32) / maxvalue
            preds = np.array(scripted_model(inputs))

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
            assert inputs.dtype in {np.dtype('uint8'), np.dtype('uint16')}, f'input type {inputs.dtype} not supported'
            maxvalue = 65535 if inputs.dtype == np.dtype('uint16') else 255
            inputs = tf.constant(inputs[None, ...], dtype=tf.float32) / maxvalue
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
    assert img.dtype in {np.dtype('uint8'), np.dtype('uint16')}, f"img.dtype: {img.dtype} not supported"
    assert (curves >= 0).all(), f'curves.min: {curves.min()}'
    assert (curves <= 1).all(), f'curves.max: {curves.max()}'
    curves = curves.reshape(3, 256)

    # Handle gray scale image
    if img.ndim == 2:
        img = img[:, :, None]
        curves = curves.mean(axis=0, keepdims=True)

    # Optionally export curves
    filename = Path('.autolevels_exported_curves.h5')
    dataset_name = '001'
    if filename.exists():
        with h5py.File(filename, 'a') as hdf_file:
            if dataset_name in hdf_file:
                # Resize the existing dataset
                existing_dataset = hdf_file[dataset_name]
                existing_size = existing_dataset.shape[0]
                new_size = existing_size + curves.shape[0]
                existing_dataset.resize(new_size, axis=0)
                existing_dataset[existing_size:] = curves
            else:
                # Create a new dataset, resizable along axis 0
                hdf_file.create_dataset(dataset_name, data=curves, maxshape=(None, 256))

    # Upsample curves for uint16 images
    if img.dtype == np.uint16:
        x_original = np.linspace(0, 1, 256)
        x_new = np.linspace(0, 1, 65536)
        curves = np.stack([np.interp(x_new, x_original, curve) for curve in curves])

    transformed = np.empty(img.shape, dtype=curves.dtype)

    # Map each channel using fancy indexing
    for i, curve in enumerate(curves):
        transformed[:, :, i] = curve[img[:, :, i]]

    # Remove channel dim from gray scale image
    if transformed.shape[2] == 1:
        transformed = transformed[:, :, 0]

    return transformed  # return as np.float32 (0, 1)
