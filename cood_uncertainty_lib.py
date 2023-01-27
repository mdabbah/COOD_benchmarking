from model_handling_utilities import get_model_name, args_dict_to_str, get_dataset_name
from utils.data_utils import load_model_results, create_dataset_metadata, save_model_results, create_data_loader, \
    load_dataset_metadata
import numpy as np

from utils.severity_estimation_utils import calc_per_class_severity, get_severity_levels_groups_of_classes


def apply_model_function_on_dataset_samples(rank, model, datasets, datasets_subsets, batch_size,
                                            num_workers, function_name, confidence_args=None):
    print(f"Running on rank {rank}.")

    # create model and move it to GPU with id rank
    model, transform = create_model_and_transforms_OOD(model, pretrained=True)

    # create the data loader.
    sampler_opt = None  # {'sampler_type': 'distributed'}
    all_data_loader = create_data_loader(datasets,
                                         ds_subsets=datasets_subsets, batch_size=batch_size,
                                         num_workers=num_workers,
                                         transform=transform)

    model = MySimpleWrapper(model.cuda(rank), model_name=model)

    with Timer(f'mcp time on {datasets_subsets} is:'):
        if function_name in ['softmax_conf', 'entropy_conf', 'max_logit_conf']:
            results = get_dataset_softmax_and_entropy_statistics(model, all_data_loader, device=rank)

        elif function_name in ['mcd_entropy', 'mutual_information', 'mcd_softmax']:
            results = get_dataset_MC_dropout_statistics(model, all_data_loader, device=rank)

        elif function_name == 'last_layer_activations':
            results = get_dataset_last_activations(model, all_data_loader, device=rank)

        elif function_name == 'odin_conf':
            results = calc_odin_confidences(model, all_data_loader, device=rank, confidence_args=confidence_args)

        elif function_name == 'embeddings':

            results = get_dataset_embeddings(model, all_data_loader, device=rank)

    del model
    # save_model_results(model_name, results, f'{tag}_{rank}')
    print(f'for rank {rank} here are the results')
    return results


def calc_model_confidences_on_datasets_samples():
    pass


def aggregate_confidences(results_list, axis=None):
    confidences = {k: [] for k in results_list[0].keys()}

    for r in results_list:
        for k, v in confidences.items():
            v.extend(r[k])

    confidences = {k: np.concatenate(v, axis=axis) for k, v in confidences.items()}
    return confidences


def get_cood_benchmarking_datasets(model, cood_dataset_info, num_severity_levels, num_id_classes, batch_size=64,
                                   num_workers=2, confidence_metric='softmax_conf', rank=0, confidence_args=None):
    assert confidence_metric in ['softmax_conf', 'entropy_conf', 'mcd_entropy', 'mutual_information',
                                 'mcd_softmax', 'odin_conf', 'max_logit_conf']

    cood_dataset_name = get_dataset_name(cood_dataset_info)  # ImageNet_20K
    create_dataset_metadata(cood_dataset_info)

    base_model_name = model_name = get_model_name(model)
    partial_tag = confidence_metric

    # in case you want to experiment with kappas that have hyper-parameters
    confidence_args_str = args_dict_to_str(confidence_args)
    confidence_file_tag = f'confidence_scores_{partial_tag}{confidence_args_str}_cood_estimation_samples'

    # part 1: estimate severity
    train_ood_confidences = load_model_results(model_name, confidence_file_tag)
    if train_ood_confidences is None:
        results = apply_model_function_on_dataset_samples(rank=rank, model=base_model_name,
                                                          datasets=[cood_dataset_name],
                                                          datasets_subsets=['train'],
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          function_name=confidence_metric,
                                                          confidence_args=confidence_args)

        print(f'we got val results list')

        results_list = [results]
        train_ood_confidences = aggregate_confidences(results_list)
        save_model_results(model_name, train_ood_confidences, confidence_file_tag)

    # part 2: create severity levels

    per_class_avg_confidence = calc_per_class_severity(train_ood_confidences, confidence_field_name=confidence_metric)

    severity_levels_info = get_severity_levels_groups_of_classes(per_class_avg_confidence,
                                                                 num_severity_levels=num_severity_levels,
                                                                 num_classes_per_group=num_id_classes)

    save_model_results(model_name, severity_levels_info,
                       f'severity_levels_info_{confidence_metric}{confidence_args_str}')

    return severity_levels_info


## one api higher
# def get_paper_results
# function to plot the graphs
# one liner for one model or multiple models
# leadership board



def get_model_cood_results(model, confidence_metric, confidence_args, cood_dataset_info,
                           id_dataset_info, severity_levels_info, levels_to_benchmark,
                           batch_size=64,
                           num_workers=2, rank=0):

    cood_dataset_name = get_dataset_name(cood_dataset_info)  # ImageNet_20K
    id_dataset_name = get_dataset_name(id_dataset_info)  # ImageNet_1K

    create_dataset_metadata(cood_dataset_info)
    create_dataset_metadata(id_dataset_name)

    model_name = get_model_name(model)
    partial_tag = confidence_metric

    # in case you want to experiment with kappas that have hyper-parameters
    confidence_args_str = args_dict_to_str(confidence_args)
    # part 3: evaluate on test
    confidence_file_tag = f'stats_{partial_tag}{confidence_args_str}_all_val'


    # get cood datasets classes
    cood_classes = severity_levels_info['severity_levels_groups'][levels_to_benchmark]

    validation_confidences = load_model_results(model_name, confidence_file_tag)
    if validation_confidences is None:
        results = apply_model_function_on_dataset_samples(rank=rank, model=model,
                                                          datasets=[id_dataset_name, cood_dataset_name],
                                                          datasets_subsets=['val', cood_classes],
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          function_name=confidence_metric,
                                                          confidence_args=confidence_args)

        print(f'we got val results list')

        results_list = [results]
        validation_confidences = aggregate_confidences(results_list)
        save_model_results(model_name, validation_confidences, confidence_file_tag)

    # part 4: evaluate the OOD performance

    num_id_classes = load_dataset_metadata(id_dataset_name)['num_classes']
    model_info = {'model_name': model_name}
    validation_confidences['is_ID'] = validation_confidences['labels'] < num_id_classes

    ood_results = calc_OOD_metrics(cood_classes + num_id_classes, validation_confidences, confidence_metric)

    model_results = log_ood_results(model_info, ood_results, f'{confidence_metric}{confidence_args_str}',
                    f'{confidence_metric}{confidence_args_str}', levels_to_benchmark)

    return model_results
