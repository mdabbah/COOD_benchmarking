import os.path
import urllib.request
import progressbar
dummy_dataset_url = 'https://github.com/mdabbah/COOD_benchmarking/releases/download/dummy_dataset/dummy_dataset.tar.gz'

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


def download_dummy_dataset(download_dir=None):
    if download_dir is None:
        download_dir = get_save_dir_from_user()
    os.makedirs(download_dir, exist_ok=True)
    targz_path = os.path.join(download_dir, 'dummy_dataset.tar.gz')
    id_path = os.path.join(download_dir, 'dummy_id')
    ood_path = os.path.join(download_dir, 'dummy_ood')

    if os.path.exists(id_path) and os.path.exists(ood_path):
        print('Dummy dataset was already found in the given directory. Skipping its download.')
        return
    print('Downloading dummy dataset')
    urllib.request.urlretrieve(dummy_dataset_url, targz_path, show_progress)

    # import mechanize
    # browser = mechanize.Browser()
    # browser.retrieve(dummy_dataset_url,
    #                  './test')

    print('Extracting dataset into ' + download_dir)
    os.system(f'tar -xzf {targz_path} -C {download_dir}')

    print('Dummy dataset downloaded successfully.')
    print('You can find the dataset at: ' + download_dir)
    print('In-distribution component of the dummy dataset path: ' + id_path)
    print('Class-out-of-distribution component of the dummy dataset path:' + ood_path)



if __name__ == '__main__':
    download_dummy_dataset()