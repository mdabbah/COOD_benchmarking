import os.path
import urllib.request
import progressbar
dummy_dataset_url = 'https://github.com/mdabbah/COOD_benchmarking/archive/refs/tags/dummy_dataset.tar.gz'


def get_save_dir_from_user():
    while True:
        path = input('please enter path destination for dummy dataset:')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return path
        if os.path.isfile(path):
            print('given path is a file please give us a valid path')
            continue
        return path

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None



def download_file(url):
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def download_dummy_dataset():
    download_dir = get_save_dir_from_user()
    targz_path = os.path.join(download_dir, 'dummy_dataset.tar.gz')
    id_path = os.path.join(download_dir, 'dummy_id')
    ood_path = os.path.join(download_dir, 'dummy_ood')

    # os.system(f'wget {dummy_dataset_url} {targz_path}')
    urllib.request.urlretrieve(dummy_dataset_url, targz_path, show_progress)

    os.system(f'tar -xzvf {targz_path}')

    print('dummy dataset downloaded successfully')
    print('you can find the dataset at:')
    print('id_dummy dataset path: ' + id_path)
    print('you can find the dataset at:' + ood_path)

    # urllib.request.urlretrieve(dummy_dataset_url, filename=targz_path)
