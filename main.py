from cood_uncertainty_lib import benchmark_model_on_cood_with_severities

if __name__ == '__main__':

    results = benchmark_model_on_cood_with_severities(model='resnet50', confidence_metric='softmax_conf',
                                                      confidence_args=None,
                                                      cood_dataset_info='default',
                                                      id_dataset_info='default', num_severity_levels=11,
                                                      levels_to_benchmark='all', batch_size=64, num_workers=2, rank=0)
