import plotly.express

from cood_uncertainty_lib import benchmark_model_on_cood_with_severities

if __name__ == '__main__':
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
    dummy_ood_dataset_info = {
        'dataset_name': 'Dummy_OOD',
        # 'images_base_folder': '.\dummy_dataset\dummy_ood',  # <path to images dir>/classname/*.(jpg|png|jpeg)
        'images_base_folder': '/media/mohammed/Elements1/exported_datasets/dummy_ood',  # <path to images dir>/classname/*.(jpg|png|jpeg)
        'test_estimation_split_percentage': 0.25
    }

    dummy_id_dataset_info = {
        'dataset_name': 'Dummy_ID',
        # 'images_base_folder': '.\dummy_dataset\dummy_id',
        'images_base_folder': '/media/mohammed/Elements1/exported_datasets/dummy_id'
    }

    results2 = benchmark_model_on_cood_with_severities(model=['resnet50', 'resnet18'],
                                                       confidence_metric=['softmax', 'entropy'],
                                                       cood_dataset_info=dummy_ood_dataset_info,
                                                       id_dataset_info=dummy_id_dataset_info)
    plotly.express.line(results2, x='severity_levels', y='cood-auroc', color='model_name-kappa')



    results = benchmark_model_on_cood_with_severities(model=['resnet50', 'resnet18'],
                                                      confidence_metric=['softmax', 'entropy'],
                                                      confidence_args=None,
                                                      cood_dataset_info='default',
                                                      id_dataset_info='default', num_severity_levels=11,
                                                      levels_to_benchmark='all', batch_size=64, num_workers=2, rank=0)

    plotly.express.line(results, x='severity_levels', y='cood-auroc', color='model_name-kappa')

    results1 = benchmark_model_on_cood_with_severities(model='resnet50', confidence_metric='softmax')

    results2 = benchmark_model_on_cood_with_severities(model='resnet50')

    #

    from torchvision.models import resnet50
    import torchvision.transforms as tvtf

    resnet50_transform = tvtf.Compose([tvtf.Resize(256), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                       tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    example_model_input = {'model_name': 'resnet50', 'nn.Module': resnet50(),
                           'nn.transforms': resnet50_transform}  # recommended
    example_model_input1 = 'resnet50'  # recommended
    example_model_input2 = {'model_name': 'resnet50'}
    example_model_input3 = resnet50()
    example_model_input4 = {'nn.Module': resnet50()}

    example_model_input5 = [example_model_input, example_model_input2, example_model_input3]

    from utils.kappa_extractors import extract_softmax_on_dataset

    example_kappa_input = extract_softmax_on_dataset
    example_kappa_input1 = {'confidence_metric_name': 'softmax',
                            'confidence_metric_callable': extract_softmax_on_dataset}  # recommended

    example_kappa_input2 = {'confidence_metric_callable': extract_softmax_on_dataset}
