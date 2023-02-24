import glob
import os
import pickle
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.custom_dataset import CustomedDataset
from utils.project_paths import get_results_base_path, get_datasets_metadata_base_path


def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not path.endswith('.pkl'):
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    obj = None
    if not path.endswith('.pkl'):
        path += '.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

    return obj


def get_dataset_num_of_classes(dataset_info_dict):
    return len(get_dataset_classes(dataset_info_dict))


def get_dataset_classes(dataset_info_dict_or_name):
    if isinstance(dataset_info_dict_or_name, str):
        dataset_name = dataset_info_dict_or_name
    else:
        dataset_name = dataset_info_dict_or_name['dataset_name']

    dataset_meta_data = load_dataset_metadata(dataset_name)
    classes = dataset_meta_data['class_names']
    return classes


def create_dataset_metadata(dataset_info_dict, is_id_dataset=False):
    dataset_name = dataset_info_dict['dataset_name']

    ds_metadata_base_path = get_datasets_metadata_base_path()
    dataset_metadata_path = os.path.join(ds_metadata_base_path, dataset_name + '_metadata.pkl')
    if os.path.exists(dataset_metadata_path):
        return

    dataset_base_folder = dataset_info_dict['images_base_folder']

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.JPEG']

    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(f'{dataset_base_folder}/*/*{extension}', recursive=True))
    if len(image_files) == 0:
        raise ValueError(f'we scanned the given path {dataset_base_folder} but couldn\'t find any images.'
                         f'are you sure you sent the correct path?')
    image_files = np.array(image_files)
    class_names = [os.path.basename(os.path.dirname(f)) for f in image_files]
    class_names, labels = np.unique(class_names, return_inverse=True)

    if is_id_dataset:

        val_idx = np.arange(len(image_files))
        train_idx = None
    else:

        percentage = dataset_info_dict['test_estimation_split_percentage']
        train_idx, val_idx = split_dataset(labels, percentage)

    meta_data = {'image_files': image_files, 'labels': labels, 'class_names': class_names, 'train_idx': train_idx,
                 'val_idx': val_idx, 'num_classes': len(class_names), 'images_base_folder': dataset_base_folder}

    save_pickle(dataset_metadata_path, meta_data)


# def get_dataset_meta


def load_dataset_metadata(dataset_name):
    """
    :param dataset_name: the dataset to load metadata for
    :return: a dictionary of metadata
    that most importantly contains instances paths and labels.
    """
    from glob import glob
    datasets_metadata_base_path = get_datasets_metadata_base_path()

    available_datasets = glob(datasets_metadata_base_path + '/*metadata.pkl')

    metadata_datasets = {os.path.basename(ds).split('_metadata.pkl')[0]: ds for ds in available_datasets}

    metadata_path = metadata_datasets.get(dataset_name)
    if metadata_path is None:
        return None
    metadata = load_pickle(metadata_path)
    return metadata


def load_ds_img_paths_and_labels(dataset_name, ds_subset=None):
    """
    :param dataset_name: dataset to load metadata for.
    :param ds_subset: the subset of the metadata (e.g. 'training', 'validation', 'test').
    (relevant for ImageNet_1K)
    :return: tuple: img_paths, labels
    """
    metadata = load_dataset_metadata(dataset_name)
    if metadata is None:
        raise ValueError('given dataset meta data does not exist.\nthis should not happen')

    # img_paths = norm_paths(dataset_name, metadata['image_files'])
    img_paths = metadata['image_files']
    labels = metadata['labels']

    if ds_subset is not None:
        if isinstance(ds_subset, str):
            subset_idx = metadata[ds_subset + '_idx']
            img_paths = img_paths[subset_idx]
            labels = labels[subset_idx]
        else:
            # assumes its severity_groups classes
            subset_idx = metadata['val_idx']
            img_paths = img_paths[subset_idx]
            labels = labels[subset_idx]

            mask = np.isin(labels, np.reshape(ds_subset, (1, -1)))
            img_paths = img_paths[mask]
            labels = labels[mask]

    return img_paths, labels


def norm_paths(img_paths, new_base_folder, split_keyword):
    img_paths = [os.path.join(new_base_folder, img.split(split_keyword)[1]) for img in img_paths]

    return np.array(img_paths)


def check_file_exists(img_path):
    if not os.path.exists(img_path):
        raise ValueError(f"Error: could not find {img_path} when scanning the given directory "
                         f"which was part od the dataset used in the paper")


def check_files_exist(image_files):
    pool = ThreadPool()  # Create a multiprocessing pool
    # Use a set for faster lookups
    existing_files = set(filter(os.path.exists, image_files))
    # Use tqdm to display progress
    for _ in tqdm(pool.imap(check_file_exists, existing_files), total=len(existing_files),
                  desc='scanning given folder for the images used in our dataset'):
        pass


def combine_datasets(dataset_names, ds_subsets):
    all_img_paths = []
    all_labels = []
    offset = 0
    for dataset_name, ds_subset in zip(dataset_names, ds_subsets):
        img_paths, labels = load_ds_img_paths_and_labels(dataset_name, ds_subset)
        classes, labels_reordered = np.unique(labels, return_inverse=True)
        labels += offset
        offset += len(classes)
        all_labels.extend(labels)

        all_img_paths.extend(img_paths)

    return np.array(all_img_paths), np.array(all_labels)


def create_data_loader(dataset_names, ds_subsets, batch_size, num_workers, shuffle=False, offset=0, num_shots=None,
                       transform=None):
    """
    :return: returns a PyTorch data loader.
    """

    if isinstance(dataset_names, list):
        img_paths, labels = combine_datasets(dataset_names, ds_subsets)

    else:
        img_paths, labels = load_ds_img_paths_and_labels(dataset_names, ds_subsets)
        _, labels = np.unique(labels, return_inverse=True)
        labels += offset

    print(f'loaded dataset will have {len(labels)} samples.')
    assert len(img_paths) == len(labels)
    subset = CustomedDataset(img_paths, labels, num_shots, transform)

    sampler = None
    batch_sampler = None

    # noinspection PyArgumentList
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_workers, sampler=sampler,
                                                batch_sampler=batch_sampler)

    return subset_loader


def save_model_results(results_subdir_name, data, data_name):
    base_folder = get_results_base_path()

    save_path = os.path.join(base_folder, results_subdir_name, data_name)
    save_pickle(save_path, data)


def load_model_results(results_subdir_name, data_name):
    base_folder = get_results_base_path()

    save_path = os.path.join(base_folder, results_subdir_name, data_name)

    data = load_pickle(save_path)
    return data


def delete_model_results(results_subdir_name, data_name):
    base_folder = get_results_base_path()

    save_path = os.path.join(base_folder, results_subdir_name, data_name)
    if os.path.exists(save_path):
        os.remove(save_path)


def check_model_results_exist(results_subdir_name, data_name):
    base_folder = get_results_base_path()

    if not data_name.endswith('.pkl'):
        data_name += '.pkl'

    save_path = os.path.join(base_folder, results_subdir_name, data_name)

    return os.path.exists(save_path)


def load_model_results_df(results_subdir_name, data_name, base_path=None):
    if base_path is None:
        base_path = get_results_base_path()

    load_path = os.path.join(base_path, results_subdir_name, data_name)
    if not os.path.exists(load_path):
        # print(f'could not find file {load_path}')
        return None
    data = pd.read_csv(load_path)
    return data


def save_model_results_df(df: pd.DataFrame, results_subdir_name, data_name, base_path=None):
    if base_path is None:
        base_path = get_results_base_path()
    save_path = os.path.join(base_path, results_subdir_name, data_name)
    df.to_csv(save_path, index=False)


def split_dataset(labels, test_percentage):
    train_idx = []
    val_idx = []
    classes = np.unique(labels)
    classes_samples = {c: [] for c in classes}
    train_percentage = 1 - test_percentage
    for s, l in enumerate(labels):
        classes_samples[l].append(s)

    for c in classes:
        class_samples = np.random.permutation(np.array(classes_samples[c]))
        num_train_samples = int(train_percentage * len(class_samples))
        # assert len(class_samples) == 200
        # assert num_train_samples == 150
        train_idx.extend(class_samples[:num_train_samples])
        val_idx.extend(class_samples[num_train_samples:])

    return np.array(train_idx), np.array(val_idx)
