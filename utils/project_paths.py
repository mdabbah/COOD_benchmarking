import os

project_dir = os.path.dirname(os.path.dirname(__file__))


def get_results_base_path():
    return os.path.join(
        project_dir, 'results'
    )


def get_paper_results_base_path():
    return os.path.join(
        project_dir, 'paper_results'
    )


def get_datasets_metadata_base_path():
    return os.path.join(
        project_dir, 'datasets_metadata'
    )
