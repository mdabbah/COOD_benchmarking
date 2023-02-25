import itertools
import os.path
import zipfile
from typing import List

import pandas as pd
import torch.nn
from tqdm import tqdm

from utils.input_handling_utilities import get_model_name, args_dict_to_str, get_dataset_name, handle_parameters, \
    handle_model_dict_input, get_kappa_name, sanity_check_confidence_input, CONFIDENCE_METRIC_INPUT_ERR_MSG, \
    sanity_model_input, MODEL_INPUT_ERR_MSG, check_and_fix_transforms
from utils.data_utils import load_model_results, create_dataset_metadata, save_model_results, create_data_loader, \
    get_dataset_num_of_classes, load_model_results_df, load_pickle, norm_paths, save_pickle
import numpy as np

from utils.kappa_dispatcher import get_confidence_function
from utils.models_wrapper import MySimpleWrapper
from utils.project_paths import get_datasets_metadata_base_path
from utils.severity_estimation_utils import calc_per_class_severity, get_severity_levels_groups_of_classes
from utils.misc import create_model_and_transforms_OOD, log_ood_results, get_default_transform_with_open, \
    aggregate_results_from_batches
from utils.uncertainty_metrics import calc_OOD_metrics


def apply_model_function_on_dataset_samples(rank, model, datasets, datasets_subsets, batch_size,
                                            num_workers, function, confidence_args=None):
    # print(f"Running on rank {rank}.")

    # create model and move it to GPU with id rank
    model_name = get_model_name(model)
    transform = None
    if isinstance(model, str):
        model, transform = create_model_and_transforms_OOD(model, pretrained=True)
    elif isinstance(model, dict):
        model, transform = handle_model_dict_input(model)
    elif isinstance(model, torch.nn.Module):
        transform = None
    else:
        raise ValueError(f'unrecognized model input form {type(model)}')
    assert isinstance(model, torch.nn.Module)

    if transform is None:
        transform = get_default_transform_with_open()

    transform = check_and_fix_transforms(transform)

    # create the data loader.
    all_data_loader = create_data_loader(datasets,
                                         ds_subsets=datasets_subsets, batch_size=batch_size,
                                         num_workers=num_workers,
                                         transform=transform)

    #####
    # if dealing with dummy dataset we prone the classification layer to include only
    # ID dummy dataset classes.
    #####

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    model = MySimpleWrapper(model.to(device), model_name=model_name, datasets=datasets)
    if isinstance(function, dict):
        function = function['confidence_metric_callable']

    function = get_confidence_function(function)
    results = function(model, all_data_loader, device=device, confidence_args=confidence_args)

    del model
    return results


def get_cood_benchmarking_datasets(model, confidence_function='softmax_conf', confidence_args=None,
                                   cood_dataset_info='default', num_severity_levels=11, num_id_classes=1000,
                                   batch_size=64, num_workers=2, rank=0, force_run=False, confidence_key='confidences',
                                   results_subdir_name=None):
    assert sanity_check_confidence_input(confidence_function), CONFIDENCE_METRIC_INPUT_ERR_MSG
    assert sanity_model_input(model), MODEL_INPUT_ERR_MSG

    confidence_args_str = args_dict_to_str(confidence_args)
    model_name = get_model_name(model)

    confidence_metric_name = get_kappa_name(confidence_function)
    cood_dataset_name = get_dataset_name(cood_dataset_info)  # ImageNet_20K
    create_dataset_metadata(cood_dataset_info)

    if results_subdir_name is None:
        results_subdir_name = os.path.join(model_name, f'{cood_dataset_name}')

    severity_levels_info_file_tag = f'severity_levels_info_n{num_severity_levels}_' \
                                    f'{confidence_metric_name}_{confidence_key}{confidence_args_str}'
    severity_levels_info = load_model_results(results_subdir_name=results_subdir_name,
                                              data_name=severity_levels_info_file_tag)
    if severity_levels_info is not None and not force_run:
        return severity_levels_info

    partial_tag = confidence_metric_name

    # in case you want to experiment with kappas that have hyper-parameters
    confidence_file_tag = f'confidence_scores_{partial_tag}{confidence_args_str}_cood_estimation_samples'

    # part 1: estimate severity
    train_ood_confidences = load_model_results(results_subdir_name, confidence_file_tag)
    if train_ood_confidences is None and not force_run:
        results = apply_model_function_on_dataset_samples(rank=rank, model=model,
                                                          datasets=[cood_dataset_name],
                                                          datasets_subsets=['train'],
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          function=confidence_function,
                                                          confidence_args=confidence_args)

        train_ood_confidences = aggregate_results_from_batches(results)
        save_model_results(results_subdir_name, train_ood_confidences, confidence_file_tag)

    # part 2: create severity levels

    per_class_avg_confidence = calc_per_class_severity(train_ood_confidences, confidence_field_name=confidence_key)

    severity_levels_info = get_severity_levels_groups_of_classes(per_class_avg_confidence,
                                                                 num_severity_levels=num_severity_levels,
                                                                 num_classes_per_group=num_id_classes)

    save_model_results(results_subdir_name, severity_levels_info, severity_levels_info_file_tag)

    return severity_levels_info


def _benchmark_list_inputs(models_list, confidence_metrics_list, *args):
    if not isinstance(models_list, list):
        models_list = [models_list]

    if not isinstance(confidence_metrics_list, list):
        confidence_metrics_list = [confidence_metrics_list]

    all_results = []
    for m, k in itertools.product(models_list, confidence_metrics_list):
        res = benchmark_model_on_cood_with_severities(m, k, *args)

        all_results.append(res)

    return pd.concat(all_results)


def benchmark_model_on_cood_with_severities(model, confidence_function='softmax', confidence_args=None,
                                            cood_dataset_info='default',
                                            id_dataset_info='default', num_severity_levels=11,
                                            levels_to_benchmark='all',
                                            batch_size=64,
                                            num_workers=2, rank=0, force_run=False, confidence_key='confidences'):
    if isinstance(model, list) or isinstance(confidence_function, list):
        return _benchmark_list_inputs(model, confidence_function, confidence_args, cood_dataset_info, id_dataset_info,
                                      num_severity_levels, levels_to_benchmark, batch_size, num_workers, rank,
                                      force_run,
                                      confidence_key)

    assert sanity_check_confidence_input(confidence_function), CONFIDENCE_METRIC_INPUT_ERR_MSG
    assert sanity_model_input(model), MODEL_INPUT_ERR_MSG

    confidence_args_str = args_dict_to_str(confidence_args)
    model_name = get_model_name(model)
    kappa_name = get_kappa_name(confidence_function)

    (cood_dataset_info, id_dataset_info) = handle_parameters(cood_dataset_info, id_dataset_info)
    cood_dataset_name = get_dataset_name(cood_dataset_info)  # ImageNet_20K
    id_dataset_name = get_dataset_name(id_dataset_info)  # ImageNet_1K

    results_file_tag = f'{kappa_name}{confidence_args_str}_n{num_severity_levels}'

    results_subdir_name = os.path.join(model_name, f'{cood_dataset_name}-{id_dataset_name}')

    model_results = load_model_results_df(results_subdir_name, f'{model_name}_{results_file_tag}.csv')
    if levels_to_benchmark == 'all':
        levels_to_benchmark = np.arange(num_severity_levels)
    if model_results is not None and not force_run:
        return model_results[model_results['severity level'].isin(levels_to_benchmark)]

    create_dataset_metadata(cood_dataset_info)
    create_dataset_metadata(id_dataset_info, is_id_dataset=True)

    num_id_classes = get_dataset_num_of_classes(id_dataset_info)

    partial_tag = kappa_name

    # in case you want to experiment with kappas that have hyperparameters
    # part 3: evaluate on test

    severity_levels_info = get_cood_benchmarking_datasets(model, confidence_function=confidence_function,
                                                          confidence_args=confidence_args,
                                                          cood_dataset_info=cood_dataset_info,
                                                          num_severity_levels=num_severity_levels,
                                                          num_id_classes=num_id_classes,
                                                          batch_size=batch_size, num_workers=num_workers, rank=rank,
                                                          force_run=force_run, confidence_key=confidence_key,
                                                          results_subdir_name=results_subdir_name)

    # get cood datasets classes
    cood_classes = severity_levels_info['severity_levels_groups']

    confidence_file_tag = f'stats_{partial_tag}{confidence_args_str}_all_val'
    validation_confidences = load_model_results(results_subdir_name, confidence_file_tag)
    if validation_confidences is None and not force_run:
        results = apply_model_function_on_dataset_samples(rank=rank, model=model,
                                                          datasets=[id_dataset_name, cood_dataset_name],
                                                          datasets_subsets=['val', 'val'],
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          function=confidence_function,
                                                          confidence_args=confidence_args)

        validation_confidences = aggregate_results_from_batches(results)
        save_model_results(results_subdir_name, validation_confidences, confidence_file_tag)

    # part 4: evaluate the OOD performance

    validation_confidences['is_ID'] = validation_confidences['labels'] < num_id_classes

    ood_results = calc_OOD_metrics(cood_classes + num_id_classes, validation_confidences, confidence_key)

    percentiles = severity_levels_info['percentiles']
    model_info = {'model_name': model_name, 'kappa': f'{kappa_name}_{confidence_key}',
                  'id dataset': id_dataset_name, 'ood dataset': cood_dataset_name,
                  'results_subdir_name': results_subdir_name}

    model_results = log_ood_results(model_info, ood_results, results_file_tag, percentiles)

    model_results = model_results[model_results['severity level'].isin(levels_to_benchmark)]

    return model_results


def get_paper_results(model_name: [str, None, List] = None,
                      confidence_function: [str, None, List] = None) -> pd.DataFrame:
    if isinstance(model_name, str):
        model_name = [model_name]

    if isinstance(confidence_function, str):
        confidence_function = [confidence_function]

    if isinstance(confidence_function, List):
        confidence_function = [f'odin_temperature-2_noise_mag-1e-05' if k == 'odin' else k
                               for k in confidence_function]

    all_results = pd.read_csv('./paper_results/all_paper_results.csv')
    if model_name is not None:
        query = '`model name` in @model_name'
        all_results = all_results.query(query).copy()

    if confidence_function is not None:
        query = '`confidence function` in @confidence_function'
        all_results = all_results.query(query).copy()

    return all_results


def get_paper_ood_dataset_info(path_to_full_imagenet21k, skip_scan=False, exclude_biologically_distinct_classes=False,
                               exclude_visually_ambiguous_objects=True):
    if exclude_biologically_distinct_classes or not exclude_visually_ambiguous_objects:
        raise ValueError('not supported yet')

    datasets_metadata_base_path = get_datasets_metadata_base_path()
    metadata_path = os.path.join(datasets_metadata_base_path, 'paper_prebuilt_metadata', 'IMAGENET_20k_METADATA.pkl')

    new_dataset_name = 'paper_default_ood_dataset_v.4.0'

    dataset_info = _fix_prebuilt_dataset_meta_data_paths(new_dataset_base_dir=path_to_full_imagenet21k,
                                                         metadata_path=metadata_path,
                                                         new_dataset_name=new_dataset_name,
                                                         stitch_keyword='fall11_whole/',
                                                         skip_scan=skip_scan)
    dataset_info['test_estimation_split_percentage'] = 0.25

    return dataset_info


def get_paper_id_dataset_info(path_to_full_imagenet1k, skip_scan=False):
    datasets_metadata_base_path = get_datasets_metadata_base_path()
    metadata_path = os.path.join(datasets_metadata_base_path, 'paper_prebuilt_metadata', 'IMAGENET_1k_val_METADATA.pkl')

    new_dataset_name = 'paper_default_id_dataset_v.4.0'

    dataset_info = _fix_prebuilt_dataset_meta_data_paths(new_dataset_base_dir=path_to_full_imagenet1k,
                                                         metadata_path=metadata_path,
                                                         new_dataset_name=new_dataset_name,
                                                         stitch_keyword='LSVRC2012_img_val/',
                                                         skip_scan=skip_scan)

    return dataset_info


def _fix_prebuilt_dataset_meta_data_paths(new_dataset_base_dir, metadata_path,
                                          new_dataset_name, stitch_keyword, skip_scan):
    datasets_metadata_base_path = get_datasets_metadata_base_path()
    if not os.path.exists(metadata_path):
        path_to_zip_file = os.path.join(datasets_metadata_base_path, 'paper_prebuilt_metadata',
                                        'datasets_metadata_v4.zip')
        directory_to_extract_to = os.path.join(datasets_metadata_base_path, 'paper_prebuilt_metadata')

        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    dataset_metadata = load_pickle(metadata_path)
    new_metadata_path = os.path.join(datasets_metadata_base_path, new_dataset_name + '_metadata.pkl')

    if os.path.exists(new_metadata_path) and skip_scan:
        # if it exists then load it and make sure the given path is still the one used
        # in the cached metadata
        new_dataset_metadata = load_pickle(new_metadata_path)
        images_base_dir = new_dataset_metadata['dataset_base_folder']
        if images_base_dir == new_dataset_base_dir:
            # no need to fix or scan
            return {'dataset_name': new_dataset_name, 'images_base_folder': new_dataset_base_dir}

    image_files = dataset_metadata['image_files']
    image_files = norm_paths(image_files, new_dataset_base_dir, stitch_keyword)

    dataset_metadata['image_files'] = image_files
    dataset_metadata['dataset_base_folder'] = new_dataset_base_dir
    dataset_metadata['class_names'] = dataset_metadata['wordnet_ids']

    if not skip_scan:
        # check_files_exist(image_files)
        for img_path in tqdm(image_files, desc='scanning given folder for the images used in our dataset'):

            if not os.path.exists(img_path):
                raise ValueError(f"Error: could not find {img_path} when scanning the given directory "
                                 f"which was part od the dataset used in the paper")

    save_pickle(new_metadata_path, dataset_metadata)
    print(f'saved a new metadata dataset at {new_metadata_path}')

    return {'dataset_name': new_dataset_name, 'images_base_folder': new_dataset_base_dir}
