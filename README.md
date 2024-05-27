# A framework for Benchmarking Class-out-of-distribution Detection
<p align="center">
  <img src="https://github.com/mdabbah/COOD_benchmarking/blob/main/degredation_graph_paper.png">
</p>
This repository is the official implementation of "A framework for Benchmarking Class-out-of-distribution Detection and its Application to ImageNet", accepted to ICLR 2023 as a notable-top-25%, by Ido Galil, Mohammed Dabbah and Ran El-Yaniv:
https://openreview.net/pdf?id=Iuubb9W6Jtk


## Usage
An in-depth guide with examples for how to use the framework is provided in example.ipynb, it is **very recommended** to start there.

This framework allows to benchmark C-OOD detection performance on any custom dataset (including ImageNet, as described in the paper) of models and their confidence function (OOD detection algorithm, i.e, ODIN, softmax, MC Dropout, etc.).

## Leaderboard
We invite you to share the performance of your models and techniques on:
https://paperswithcode.com/sota/classification-on-imagenet-c-ood-class-out-of

## Instructions for downloading and extracting our filtered version of ImageNet-21k
If you want to use our filtered version of the ImageNet-21k dataset, you can either download the entire ImageNet-21k dataset and follow our example in the "Using our benchmark with ImageNet" chapter of example.ipynb to transform it into our filtered version, or you can simply follow the instructions below to only download our filtered version:

To download our filtered dataset, please access the file parts at the following link:
https://technionmail-my.sharepoint.com/:f:/g/personal/mdabbah_alumni_technion_ac_il/Enq4_ikDQzVIlsMlc3toIvsBMfukjl_ocs2AKsDirAa00A

Concatenate the dataset zip parts into a single tar file and extract it to a directory of your choice using the following commands:

```
cat dataset.parta* > dataset.tar                     ## This command concatenates the dataset parts together.
tar -xzvf dataset.tar.gz -C </path/to/directory/>    ## This command extracts the dataset to a given directory.
unzip missing_files.zip -d </path/to/directory>      ## This command adds 46 missing files from our original tar file.
```

After executing these commands, you will find a directory called "imagenet_11k" in the location specified by `</path/to/directory/>`.

After downloading and extracting the dataset, use our method `get_paper_ood_dataset_info` to build a dataset metadata object that's used for dataset loading in our implementation (see example.ipynb for usage examples). 
Make sure to pass the path to the dataset as </path/to/directory>/imagenet_11k. 
You can use the skip_scan parameter to check for missing files in your copy by setting it to 'False'.

We recommend developing and testing in a Linux environment, as we are aware that Windows Defender might delete some images (the ones from 'missing_files.zip'), thinking they are a threat.

For the ID dataset, we used the ImageNet 1K validation set. Please download it from the official ImageNet website after creating an account and logging in at:
https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

After downloading the `ILSVRC2012_img_val.tar` file, decompress it to a directory of your choice using the following command:

```
tar -xzvf ILSVRC2012_img_val.tar -C </path/to/destination_id/> 
```

After decompression, you will find a directory called "ILSVRC2012_img_val" in your directory.

To use our method `get_paper_id_dataset_info`, pass the path `</path/to/destination_id/>/ILSVRC2012_img_val`. In that method, you can also use the `skip_scan` flag to check for missing files in your copy of ImageNet's validation set.

Please contact us if you find images missing.


## Upcoming features
We plan to add a few more features in the following updates, including:

1. Including \ excluding “biologically distinct” classes (viceroy butterflies vs. monarch butterflies etc.) and “visually ambiguous objects” (ping pong balls vs. cue balls etc.). More information about this is provided in Section 4 of the paper.
