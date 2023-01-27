from typing import Dict

import torch.nn

from utils.misc import create_model_and_transforms_OOD


def get_model_name(model):
    if isinstance(model, str):
        return model
    elif 'model_name' in model:
        return model['model_name']
    elif hasattr(model, 'model_name'):
        return model.__getattr__('model_name')
    return 'default_model_results_dir'


def get_kappa_name(confidence_metric):
    if isinstance(confidence_metric, str):
        return confidence_metric
    elif 'confidence_metric_name' in confidence_metric:
        return confidence_metric['confidence_metric_name']
    elif hasattr(confidence_metric, 'confidence_metric_name'):
        return confidence_metric.__getattr__('confidence_metric_name')
    return 'given_kappa'


def get_dataset_name(dataset_info_dict):
    return dataset_info_dict['dataset_name']


default_ood_dataset_info = {
    'dataset_name': 'ImageNet_20K',
    'images_base_folder': '<path to images>',
    'percentage': 0.25
}

default_id_dataset_info = {
    'dataset_name': 'ImageNet_1K',
    'images_base_folder': '<path to images>',
}


def handle_parameters(cood_dataset_info, id_dataset_info):
    if cood_dataset_info == 'default':
        cood_dataset_info = default_ood_dataset_info

    if id_dataset_info == 'default':
        id_dataset_info = default_ood_dataset_info

    return cood_dataset_info, id_dataset_info


def handle_model_dict_input(model_dict: Dict):
    model = model_dict.get('nn.module', None)

    transforms = model_dict.get('nn.transforms', None)
    model_name = model_dict.get('model_name', None)
    if model is None and model_name is not None:
        model, t = create_model_and_transforms_OOD(model_name, pretrained=True)
        if transforms is None:
            return model, t

    return model, transforms


def sanity_check_confidence_input(confidence_metric):
    if isinstance(confidence_metric, str) and confidence_metric in ['softmax_conf', 'entropy_conf', 'mcd_entropy',
                                                                    'mutual_information',
                                                                    'mcd_softmax', 'odin_conf', 'max_logit_conf']:
        return True

    if callable(confidence_metric):
        return True

    return isinstance(confidence_metric, dict) and callable(confidence_metric.get('confidence_metric_callable'))


CONFIDENCE_METRIC_INPUT_ERR_MSG = "confidence metric input needs to be either a callble fuction or a string or a dict " \
                                  "with a key confidence_metric_callable with the value being a callable.\n" \
                                  "if the input is a string it needs to be one of ['softmax_conf', 'entropy_conf', 'mcd_entropy',\
                                                                    'mutual_information',\
                                                                    'mcd_softmax', 'odin_conf', 'max_logit_conf']"


def sanity_model_input(model):
    if isinstance(model, str):
        return True

    if isinstance(model, torch.nn.Module):
        return True

    if isinstance(model, dict):
        return isinstance(model.get('nn.Module'), torch.nn.Module) or \
               isinstance(model.get('model_name'), str)


MODEL_INPUT_ERR_MSG = "model needs to be either a string or nn.Module instance or a dictionary with keys 'model_name'" \
                      "(string) or 'nn.Module' (nn.Module)"

def args_dict_to_str(args_dict):
    if args_dict is None or not isinstance(args_dict, dict):
        return ''

    args_dict_str = ''
    for k, v in args_dict.items():
        args_dict_str += f'_{k}-{v}'

    return args_dict_str
