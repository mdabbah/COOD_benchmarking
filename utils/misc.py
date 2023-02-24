import os
import sys

import pandas as pd
import numpy as np
import timm
import torch
# from timm_lib.timm.data import resolve_data_config, create_transform
from timm.data import resolve_data_config, create_transform
import utils.old_timm_lib
from utils.old_timm_lib.timm.data import resolve_data_config as old_resolve_data_config
from utils.old_timm_lib.timm.data import create_transform as old_create_transform
# from timm.data import resolve_data_config, create_transform
import clip
# To use clip:
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git

from utils.custom_dataset import get_open_img_transforms
from utils.data_utils import load_model_results, save_model_results, get_results_base_path
from utils.log_utils import Logger

import torchvision.transforms as tvtf

import timm.models.resnetv2 as timm_bit
import timm.models.resnet as timm_resnet
import torchvision.models as torchvision_models


def normalize(a):
    a = np.array(a)
    a = (a - np.mean(a)) / np.std(a)
    return a


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def args_dict_to_str(args_dict):
    if args_dict is None or not isinstance(args_dict, dict):
        return ''

    args_dict_str = ''
    for k, v in args_dict.items():
        args_dict_str += f'_{k}-{v}'

    return args_dict_str


def get_fc_layer(model):
    """
    the worst thing i've ever written :)
    :param model: trained model
    :return: the weights of the classification layer in numpy
    """
    if hasattr(model, 'fc'):
        if isinstance(model.fc, torch.nn.Linear):
            return model.fc.weight.detach().clone().cpu().numpy()
        if isinstance(model.fc, torch.nn.Conv2d) and model.fc.kernel_size == (1, 1):
            return model.fc.weight.detach().clone().cpu().numpy()

    if hasattr(model, 'last_linear'):
        if isinstance(model.last_linear, torch.nn.Linear):
            return model.last_linear.weight.detach().clone().cpu().numpy()

    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Linear):
            return model.classifier.weight.detach().clone().cpu().numpy()

        if isinstance(model.classifier, torch.nn.Conv2d) and model.classifier.kernel_size == (1, 1):
            return model.classifier.weight.detach().clone().cpu().numpy()

        if isinstance(model.classifier, torch.nn.Sequential):
            if isinstance(model, torchvision_models.SqueezeNet):
                return model.classifier[1].weight.detach().clone().cpu().numpy()

            return model.classifier[-1].weight.detach().clone().cpu().numpy()

        if hasattr(model.classifier, 'fc'):
            if isinstance(model.classifier.fc, torch.nn.Linear):
                return model.classifier.fc.detach().clone().cpu().numpy()

    if hasattr(model, 'head'):
        if isinstance(model.head, torch.nn.Linear):
            return model.head.weight.detach().clone().cpu().numpy()
        if hasattr(model.head, 'fc'):

            if isinstance(model.head.fc, torch.nn.Linear):
                return model.head.fc.detach().clone().cpu().numpy()

            if isinstance(model.head.fc, torch.nn.Conv2d) and model.head.fc.kernel_size == (1, 1):
                return model.head.fc.weight.detach().clone().cpu().numpy()

        if hasattr(model.head, 'l'):
            if isinstance(model.head.l, torch.nn.Linear):
                return model.head.l.detach().clone().cpu().numpy()

    if hasattr(model, 'classif'):
        if isinstance(model.classif, torch.nn.Linear):
            return model.classif.weight.detach().clone().cpu().numpy()

    return False


def get_timm_transforms(model):
    config = resolve_data_config({}, model=model)
    open_img_transforms = get_open_img_transforms()
    transform = tvtf.Compose([open_img_transforms, create_transform(**config)])
    return transform


def log_ood_results(model_info, ood_results, results_file_tag, percentiles):
    model_results = pd.DataFrame(ood_results)
    model_results['model name'] = model_info['model_name']
    model_results['confidence function'] = model_info['kappa']
    model_results['percentile'] = percentiles
    model_results['severity level'] = np.arange(len(percentiles))
    model_results['model name - confidence function'] = model_info['model_name'] + '-' + model_info['kappa']

    model_results['id dataset'] = model_info['id dataset']
    model_results['ood dataset'] = model_info['ood dataset']


    model_name = model_info['model_name']
    results_subdir_name = model_info['results_subdir_name']

    base_folder = get_results_base_path()
    model_dir = os.path.join(base_folder, results_subdir_name)
    private_log_path = os.path.join(model_dir, f'{model_name}_{results_file_tag}.csv')
    model_results.to_csv(private_log_path, index=False)

    return model_results


def get_embedding_size(model_name):
    # model = timm.create_model(model_name, pretrained=False)
    # return model.num_features
    # turn on when you finally add torchvision integration
    try:
        model = create_model_and_transforms(model_name, False)[0]
        try:
            return model.num_features
        except Exception:
            return get_fc_layer(model).shape[1]
    except:
        return -1


def translate_model_name(model_name, to_our_convention=False):
    """"
    translates from timm convention to our convention or vise-versa depending on to_torchvision value
    our convention adds '_torchvision' suffix to  torchvision models and removes 'tv_' prefix if exists.
    """
    torchvision_duplicated_models_in_timms = ['resnext101_32x8d', 'tv_densenet121', 'tv_resnet101', 'tv_resnet152',
                                              'tv_resnet34', 'tv_resnet50', 'tv_resnext50_32x4d', 'vgg11', 'vgg11_bn',
                                              'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                                              'wide_resnet101_2', 'densenet169', 'densenet201', 'densenet161',
                                              'inception_v3', 'resnet18']
    if to_our_convention:
        if model_name in torchvision_duplicated_models_in_timms:
            return model_name.replace('tv_', '') + '_torchvision'
        return model_name
    else:
        if model_name.replace('_torchvision', '') in torchvision_duplicated_models_in_timms:
            return model_name.replace('_torchvision', '')
        if 'tv_' + model_name.replace('_torchvision', '') in torchvision_duplicated_models_in_timms:
            return 'tv_' + model_name.replace('_torchvision', '')
        return model_name


def translate_models_names(models_names, to_our_convention=False):
    models_names_translated = [translate_model_name(model_name, to_our_convention) for model_name in models_names]
    return models_names_translated


def _create_resnetv2_distilled_160_from224teacher(variant, pretrained=True, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    return timm_bit.build_model_with_cfg(
        timm_bit.ResNetV2, variant, pretrained,
        default_cfg=timm_bit._cfg(
            url='https://storage.googleapis.com/bit_models/distill/R50x1_160.npz',
            input_size=(3, 160, 160), interpolation='bicubic'),
        feature_cfg=feature_cfg,
        pretrained_custom_load=True,
        **kwargs)


def _create_resnet50_pruned(variant, pretrained, **kwargs):
    assert variant in [70, 83, 85]
    cfg = timm_resnet._cfg(
        url=f'https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_{variant}_state_dict.pth')
    return timm_resnet.build_model_with_cfg(
        timm_resnet.ResNet, variant='resnet50', pretrained=pretrained,
        default_cfg=cfg,
        **kwargs)


default_transform = tvtf.Compose([tvtf.Resize(256), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                  tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_default_transform_with_open():
    transforms_ = add_open_transforms(default_transform)
    return transforms_


def add_open_transforms(transforms):
    open_img_transforms = get_open_img_transforms()
    transforms = tvtf.Compose([open_img_transforms, transforms])
    return transforms


def create_model_and_transforms_OOD(model_name, pretrained=True):
    model, transforms_ = create_model_and_transforms(model_name, pretrained)
    transforms_ = add_open_transforms(transforms_)

    return model, transforms_


def create_model_and_transforms(model_name, pretrained=True, models_dir='./timmResNets',
                                weights_generic_name='model_best.pth.tar'):
    if '_torchvision' in model_name:
        pretrained_str = f'{pretrained}'

        architecture = model_name.replace('_torchvision', '')
        architecture = architecture.replace('_mcd', '')
        if 'MCdropout' in model_name:
            architecture = architecture.replace('_MCdropout', '')  # Since it's the same torchvision model
        if '_quantized' in model_name:
            architecture = architecture.replace('_quantized', '')
            model = eval(
                f'torchvision_models.quantization.' + architecture + f'(pretrained={pretrained_str}, quantize=True).eval().cuda()')
        else:
            model = eval(f'torchvision_models.' + architecture + f'(pretrained={pretrained_str}).eval().cuda()')

        if architecture == 'inception_v3':
            transform = tvtf.Compose([tvtf.Resize(342), tvtf.CenterCrop(299), tvtf.ToTensor(),
                                      tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif 'convnext' in architecture:
            resize = 232
            if architecture == 'convnext_small':
                resize = 230
            elif architecture == 'convnext_tiny':
                resize = 236
            transform = tvtf.Compose([tvtf.Resize(resize), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                      tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = default_transform
    elif model_name == 'resnetv2_50x1_bit_distilled_160_from224teacher':
        model = _create_resnetv2_distilled_160_from224teacher('resnetv2_50x1_bit_distilled_160_from224teacher',
                                                              pretrained=pretrained, stem_type='fixed',
                                                              conv_layer=timm_bit.partial(timm_bit.StdConv2d, eps=1e-8),
                                                              layers=[3, 4, 6, 3], width_factor=1).eval().cuda()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    elif 'CLIP' in model_name:
        architecture = model_name.replace('CLIP_', '')
        architecture = architecture.replace('~', '/')
        model, transform = clip.load(architecture, device="cuda")
        from utils.clip_imagenet_classes import ImageNetClip
        model = ImageNetClip(model, preprocess=transform, linear_probe=False, name=model_name)
    elif 'facebookSWAG' in model_name:
        architecture = model_name.replace('_facebookSWAG', '')
        model = torch.hub.load("facebookresearch/swag", model=architecture).eval().cuda()
        resize = 384
        if ('vit_l16_in1k' in model_name) or ('vit_h14_in1k' in model_name):
            resize = 512
        transform = tvtf.Compose(
            [tvtf.Resize(resize, interpolation=tvtf.InterpolationMode.BICUBIC), tvtf.CenterCrop(resize),
             tvtf.ToTensor(), tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif 'vit' in model_name and 'original' in model_name:
        architecture = model_name.replace('_original', '')
        model = utils.old_timm_lib.timm.models.create_model(architecture, pretrained=pretrained).eval().cuda()
        # Creating the model specific data transformation
        config = old_resolve_data_config({}, model=model)
        transform = old_create_transform(**config)
        print(f'for {model_name} we gave dropout rate of 0.1')
    else:
        architecture = model_name
        model = timm.create_model(architecture, pretrained=pretrained).eval()
        # Creating the model specific data transformation
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    return model, transform


def to_human_format_str(num):
    if num is None:
        return None
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def MC_Dropout_Pass(x, model, dropout_iterations=30, classification=True):
    # MC Dropout work

    if hasattr(model.module, 'forward_mc_dropout'):
        predictions = model.module.forward_mc_dropout(x, dropout_iterations)
    else:
        predictions = torch.empty((0, x.shape[0], 1000), device='cuda')  # 0, n_classes
        for i in range(dropout_iterations):
            output = model(x)
            output = torch.softmax(output, dim=1)
            predictions = torch.vstack((predictions, output.unsqueeze(0)))

    # Calculating mean across multiple MCD forward passes
    mean = torch.mean(predictions, dim=0)  # shape (n_samples, n_classes)??
    label_predictions = mean.max(1)[1]
    output_mean = torch.mean(predictions, dim=0, keepdim=True)
    if not classification:
        assert False  ## our library currently deals with classification

        prediction_variance = torch.gather(output_variance, -1, label_predictions.unsqueeze(-1)).squeeze(1)
        return {'prediction_variance': -prediction_variance, 'label_predictions': label_predictions}

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -torch.sum(mean * torch.log(mean + epsilon), dim=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - torch.mean(torch.sum(-predictions * torch.log(predictions + epsilon), dim=-1),
                                       dim=0)  # shape (n_samples,)

    return {'entropy_conf': -entropy, 'label_predictions': label_predictions,
            'mutual_information': mutual_info, 'mean_p': mean}  # return all


def aggregate_results_from_batches(results, axis=None):
    confidences = {k: np.concatenate(v, axis=axis) for k, v in results.items()}
    return confidences
