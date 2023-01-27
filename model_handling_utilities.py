
def get_model_name(model):
    if isinstance(model, str):
        return model
    elif 'model_name' in model:
        return model['model_name']
    return 'model_results_dir'


def get_dataset_name(dataset_info_dict):
    return dataset_info_dict['dataset_name']


def args_dict_to_str(args_dict):
    if args_dict is None or not isinstance(args_dict, dict):
        return ''

    args_dict_str = ''
    for k, v in args_dict.items():
        args_dict_str += f'_{k}-{v}'

    return args_dict_str
