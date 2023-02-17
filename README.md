# A framework for Benchmarking Class-out-of-distribution Detection
<p align="center">
  <img src="https://github.com/mdabbah/COOD_benchmarking/blob/main/degredation_graph_paper.png">
</p>
This repository is the official implementation of "A framework for Benchmarking Class-out-of-distribution Detection and its Application to ImageNet", accepted to ICLR 2023 as a notable-top-25%, by Ido Galil, Mohammed Dabbah and Ran El-Yaniv:
https://openreview.net/pdf?id=Iuubb9W6Jtk


## Usage
An in-depth guide with examples for how to use the framework is provided in example.ipynb, it is **very recommended** to start there.

This framework allows to benchmark C-OOD detection performance on any custom dataset (including ImageNet, as described in the paper) of models and their confidence function (OOD detection algorithm, i.e, ODIN, softmax, MC Dropout, etc.).

## Upcoming features
We plan to add a few more features in the following updates, including:

1. A download link to our filtered subset of ImageNet-21k used for benchmarking in the paper. This will save users the need to download the entire ImageNet-21k first.

2. Including \ excluding “biologically different” classes (viceroy butterflies vs. monarch butterflies etc.) and inanimate objects defined to be used differently (ping pong balls vs. cue balls etc.). More information about this is provided in Section 4 of the paper.
