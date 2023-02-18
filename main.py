import plotly

from cood_uncertainty_lib import benchmark_model_on_cood_with_severities, get_paper_results

if __name__ == '__main__':

    resnet50 = get_paper_results('resnet50', 'softmax')
    multiple_models = get_paper_results(['resnet50', 'resnet18'], 'softmax')
    multiple_models_kappas = get_paper_results(['resnet50', 'resnet18'], ['softmax', 'odin'])
    all_models_paper_results = get_paper_results()
    # results2 = benchmark_model_on_cood_with_severities(model='resnet50')
    # plotly.express.line(results2, x='severity_levels', y='cood-auroc', color='model_name-kappa')

    # example_ood_dataset_info = {
    #     'dataset_name': 'ImageNet_20K',
    #     'images_base_folder': '<path to images dir>',  # <path to images dir>/classname/*.(jpg|png|jpeg)
    #     'test_estimation_split_percentage': 0.25
    # }
    # example_id_dataset_info = {
    #     'dataset_name': 'ImageNet_1K',
    #     'images_base_folder': '<path to images dir>',
    # }

    from download_dummy_dataset import download_dummy_dataset
    download_dummy_dataset('../test_COOD')

    dummy_ood_dataset_info = {
        'dataset_name': 'Dummy_OOD',
        'images_base_folder': '..\dummy_dataset\dummy_ood',  # <path to images dir>/classname/*.(jpg|png|jpeg)
        # 'images_base_folder': '/media/mohammed/Elements1/exported_datasets/dummy_ood',  # <path to images dir>/classname/*.(jpg|png|jpeg)
        'test_estimation_split_percentage': 0.25
    }

    dummy_id_dataset_info = {
        'dataset_name': 'Dummy_ID',
        'images_base_folder': '..\dummy_dataset\dummy_id',
        # 'images_base_folder': '/media/mohammed/Elements1/exported_datasets/dummy_id'
    }

    from utils.confidence_functions import extract_softmax_on_dataset

    custom_confidence_function = {'confidence_metric_name': 'softmax_response',
                                  'confidence_metric_callable': extract_softmax_on_dataset}
    results = benchmark_model_on_cood_with_severities(model='resnet18',
                                                      confidence_function=custom_confidence_function,
                                                      cood_dataset_info=dummy_ood_dataset_info,
                                                      id_dataset_info=dummy_id_dataset_info)
    plotly.express.line(results, x='severity_level', y='ood-auroc', color='model_name-kappa')

    import torchvision.transforms as tvtf
    from torchvision.models import resnet50
    resnet50_transform = tvtf.Compose([tvtf.Resize(256), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                       tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    example_model_input = {'model_name': 'resnet50', 'model': resnet50(),
                           'transforms': resnet50_transform}  # recommended

    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = mobilenet_v3_small(weights)
    # Note to mohammed: the problem persists even if instead of transforms you'd use resnet50_transform for example
    example_model_input = {'model_name': 'mobilenet_v3_small', 'model': model, 'transforms': transforms}
    results = benchmark_model_on_cood_with_severities(model=example_model_input,
                                                      confidence_function='softmax',
                                                      cood_dataset_info=dummy_ood_dataset_info,
                                                      id_dataset_info=dummy_id_dataset_info)
    plotly.express.line(results, x='severity_level', y='ood-auroc', color='model_name-kappa')

    # results2 = benchmark_model_on_cood_with_severities(model=['resnet50', 'resnet18'],
    #                                                    confidence_metric=['softmax', 'entropy'],
    #                                                    cood_dataset_info=dummy_ood_dataset_info,
    #                                                    id_dataset_info=dummy_id_dataset_info)
    # plotly.express.line(results2, x='severity_level', y='ood-auroc', color='model_name-kappa')


    # results = benchmark_model_on_cood_with_severities(model=['resnet50', 'resnet18'],
    #                                                   confidence_metric=['softmax', 'entropy'],
    #                                                   confidence_args=None,
    #                                                   cood_dataset_info='default',
    #                                                   id_dataset_info='default', num_severity_levels=11,
    #                                                   levels_to_benchmark='all', batch_size=64, num_workers=2, rank=0)
    #
    # plotly.express.line(results, x='severity_levels', y='cood-auroc', color='model_name-kappa')
    #
    # results1 = benchmark_model_on_cood_with_severities(model='resnet50', confidence_metric='softmax')
    #
    # results2 = benchmark_model_on_cood_with_severities(model='resnet50')
    #
    # #
    #
    #
    # example_model_input = {'model_name': 'resnet50', 'nn.Module': resnet50(),
    #                        'nn.transforms': resnet50_transform}  # recommended
    # example_model_input1 = 'resnet50'  # recommended
    # example_model_input2 = {'model_name': 'resnet50'}
    # example_model_input3 = resnet50()
    # example_model_input4 = {'nn.Module': resnet50()}
    #
    # example_model_input5 = [example_model_input, example_model_input2, example_model_input3]
    #
    # from utils.confidence_functions import extract_softmax_on_dataset
    #
    # example_kappa_input = extract_softmax_on_dataset
    # example_kappa_input1 = {'confidence_metric_name': 'softmax',
    #                         'confidence_metric_callable': extract_softmax_on_dataset}  # recommended
    #
    # example_kappa_input2 = {'confidence_metric_callable': extract_softmax_on_dataset}
