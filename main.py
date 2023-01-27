from cood_uncertainty_lib import benchmark_model_on_cood_with_severities

if __name__ == '__main__':
    results = benchmark_model_on_cood_with_severities(model='resnet50', confidence_metric='softmax_conf',
                                                      confidence_args=None,
                                                      cood_dataset_info='default',
                                                      id_dataset_info='default', num_severity_levels=11,
                                                      levels_to_benchmark='all', batch_size=64, num_workers=2, rank=0)

    from torchvision.models import resnet50
    import torchvision.transforms as tvtf

    resnet50_transform = tvtf.Compose([tvtf.Resize(256), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                       tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    example_model_input = {'model_name': 'resnet50', 'nn.Module': resnet50(),
                           'nn.transforms': resnet50_transform}  # recommended
    example_model_input1 = resnet50()
    example_model_input2 = {'model_name': 'resnet50'}
    example_model_input3 = {'nn.Module': resnet50()}

    from utils.kappa_extractors import extract_softmax_signals_on_dataset

    example_kappa_input = extract_softmax_signals_on_dataset
    example_kappa_input1 = {'confidence_metric_name': 'softmax_conf',
                            'confidence_metric_callable': extract_softmax_signals_on_dataset}  # recommended

    example_kappa_input2 = {'confidence_metric_callable': extract_softmax_signals_on_dataset}
