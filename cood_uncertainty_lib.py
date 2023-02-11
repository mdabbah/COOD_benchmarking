import itertools

import pandas as pd
import torch.nn

from utils.input_handling_utilities import get_model_name, args_dict_to_str, get_dataset_name, handle_parameters, \
    handle_model_dict_input, get_kappa_name, sanity_check_confidence_input, CONFIDENCE_METRIC_INPUT_ERR_MSG, \
    sanity_model_input, MODEL_INPUT_ERR_MSG
from utils.data_utils import load_model_results, create_dataset_metadata, save_model_results, create_data_loader, \
    get_dataset_num_of_classes, load_model_results_df
import numpy as np

from utils.kappa_dispatcher import get_confidence_function
from utils.kappa_extractors import extract_softmax_signals_on_dataset, extract_MC_dropout_signals_on_dataset, \
    get_dataset_last_activations, extract_odin_confidences_on_dataset, get_dataset_embeddings, calc_OOD_metrics
from utils.log_utils import Timer
from utils.models_wrapper import MySimpleWrapper
from utils.severity_estimation_utils import calc_per_class_severity, get_severity_levels_groups_of_classes
from utils.misc import create_model_and_transforms_OOD, log_ood_results, default_transform


def apply_model_function_on_dataset_samples(rank, model, datasets, datasets_subsets, batch_size,
                                            num_workers, function, confidence_args=None):
    print(f"Running on rank {rank}.")

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
        transform = default_transform

    # create the data loader.
    all_data_loader = create_data_loader(datasets,
                                         ds_subsets=datasets_subsets, batch_size=batch_size,
                                         num_workers=num_workers,
                                         transform=transform)

    #####
    # if dealing with dummy dataset we prone the classification layer to include only
    # ID dummy dataset classes.
    #####

    model = MySimpleWrapper(model.cuda(rank), model_name=model_name, datasets=datasets)
    if isinstance(function, dict):
        function = function['confidence_metric_callable']

    function = get_confidence_function(function)
    with Timer(f'time on {datasets_subsets} is:'):
            results = function(model, all_data_loader, device=rank, confidence_args=confidence_args)

    del model
    return results


def aggregate_results_from_batches(results, axis=None):

    confidences = {k: np.concatenate(v, axis=axis) for k, v in results.items()}
    return confidences


def get_cood_benchmarking_datasets(model, confidence_metric='softmax_conf', confidence_args=None,
                                   cood_dataset_info='default', num_severity_levels=11, num_id_classes=1000,
                                   batch_size=64, num_workers=2, rank=0, force_run=False, confidence_key='confidences'):
    assert sanity_check_confidence_input(confidence_metric), CONFIDENCE_METRIC_INPUT_ERR_MSG
    assert sanity_model_input(confidence_metric), MODEL_INPUT_ERR_MSG

    confidence_args_str = args_dict_to_str(confidence_args)
    model_name = get_model_name(model)

    confidence_metric_name = get_kappa_name(confidence_metric)

    severity_levels_info_file_tag = f'severity_levels_info_n{num_severity_levels}_' \
                                    f'{confidence_metric_name}_{confidence_key}{confidence_args_str}'
    severity_levels_info = load_model_results(model_name=model_name, data_name=severity_levels_info_file_tag)
    if severity_levels_info is not None and not force_run:
        return severity_levels_info

    cood_dataset_name = get_dataset_name(cood_dataset_info)  # ImageNet_20K
    create_dataset_metadata(cood_dataset_info)

    partial_tag = confidence_metric_name

    # in case you want to experiment with kappas that have hyper-parameters
    confidence_file_tag = f'confidence_scores_{partial_tag}{confidence_args_str}_cood_estimation_samples'

    # part 1: estimate severity
    train_ood_confidences = load_model_results(model_name, confidence_file_tag)
    if train_ood_confidences is None and not force_run:
        results = apply_model_function_on_dataset_samples(rank=rank, model=model,
                                                          datasets=[cood_dataset_name],
                                                          datasets_subsets=['train'],
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          function=confidence_metric,
                                                          confidence_args=confidence_args)

        train_ood_confidences = aggregate_results_from_batches(results)
        save_model_results(model_name, train_ood_confidences, confidence_file_tag)

    # part 2: create severity levels

    per_class_avg_confidence = calc_per_class_severity(train_ood_confidences, confidence_field_name=confidence_key)

    severity_levels_info = get_severity_levels_groups_of_classes(per_class_avg_confidence,
                                                                 num_severity_levels=num_severity_levels,
                                                                 num_classes_per_group=num_id_classes)

    save_model_results(model_name, severity_levels_info, severity_levels_info_file_tag)

    return severity_levels_info


## one api higher
# def get_paper_results
# function to plot the graphs
# one liner for one model or multiple models
# leadership board

def benchmark_list_inputs(models_list, confidence_metrics_list, *args):
    if not isinstance(models_list, list):
        models_list = [models_list]

    if not isinstance(confidence_metrics_list, list):
        confidence_metrics_list = [confidence_metrics_list]

    all_results = []
    for m, k in itertools.product(models_list, confidence_metrics_list):
        res = benchmark_model_on_cood_with_severities(m, k, *args)

        all_results.append(res)

    return pd.concat(all_results)


def benchmark_model_on_cood_with_severities(model, confidence_metric='softmax', confidence_args=None,
                                            cood_dataset_info='default',
                                            id_dataset_info='default', num_severity_levels=11,
                                            levels_to_benchmark='all',
                                            batch_size=64,
                                            num_workers=2, rank=0, force_run=False, confidence_key='confidences'):
    if isinstance(model, list) or isinstance(confidence_metric, list):
        return benchmark_list_inputs(model, confidence_metric, confidence_args, cood_dataset_info, id_dataset_info,
                                     num_severity_levels, levels_to_benchmark, batch_size, num_workers, rank, force_run,
                                     confidence_key)

    assert sanity_check_confidence_input(confidence_metric), CONFIDENCE_METRIC_INPUT_ERR_MSG
    assert sanity_model_input(confidence_metric), MODEL_INPUT_ERR_MSG

    confidence_args_str = args_dict_to_str(confidence_args)
    model_name = get_model_name(model)

    kappa_name = get_kappa_name(confidence_metric)

    results_file_tag = f'{kappa_name}{confidence_args_str}_n{num_severity_levels}'

    model_results = load_model_results_df(model_name, results_file_tag)
    if model_results is not None and not force_run:
        return model_results[model_results.severity_levels.isin(levels_to_benchmark)]

    (cood_dataset_info, id_dataset_info) = handle_parameters(cood_dataset_info, id_dataset_info)
    cood_dataset_name = get_dataset_name(cood_dataset_info)  # ImageNet_20K
    id_dataset_name = get_dataset_name(id_dataset_info)  # ImageNet_1K

    create_dataset_metadata(cood_dataset_info)
    create_dataset_metadata(id_dataset_info, is_id_dataset=True)

    num_id_classes = get_dataset_num_of_classes(id_dataset_info)

    partial_tag = kappa_name

    # in case you want to experiment with kappas that have hyperparameters
    # part 3: evaluate on test

    severity_levels_info = get_cood_benchmarking_datasets(model, confidence_metric=confidence_metric,
                                                          confidence_args=confidence_args,
                                                          cood_dataset_info=cood_dataset_info,
                                                          num_severity_levels=num_severity_levels,
                                                          num_id_classes=num_id_classes,
                                                          batch_size=batch_size, num_workers=num_workers, rank=rank,
                                                          force_run=force_run, confidence_key=confidence_key)

    if levels_to_benchmark == 'all':
        levels_to_benchmark = np.arange(num_severity_levels)

    # get cood datasets classes
    cood_classes = severity_levels_info['severity_levels_groups']

    confidence_file_tag = f'stats_{partial_tag}{confidence_args_str}_all_val'
    validation_confidences = load_model_results(model_name, confidence_file_tag)
    if validation_confidences is None and not force_run:
        results = apply_model_function_on_dataset_samples(rank=rank, model=model,
                                                          datasets=[id_dataset_name, cood_dataset_name],
                                                          datasets_subsets=['val', 'val'],
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          function=confidence_metric,
                                                          confidence_args=confidence_args)

        validation_confidences = aggregate_results_from_batches(results)
        save_model_results(model_name, validation_confidences, confidence_file_tag)

    # part 4: evaluate the OOD performance

    validation_confidences['is_ID'] = validation_confidences['labels'] < num_id_classes

    ood_results = calc_OOD_metrics(cood_classes + num_id_classes, validation_confidences, confidence_key)

    percentiles = severity_levels_info['percentiles']
    model_info = {'model_name': model_name, 'kappa': f'{kappa_name}_{confidence_key}'}

    model_results = log_ood_results(model_info, ood_results, results_file_tag, percentiles)

    model_results = model_results[model_results['severity_level'].isin(levels_to_benchmark)]

    return model_results
