import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import torch.nn as nn

from utils.data_utils import get_dataset_classes
from utils.imagenet1K_synsets import class_to_idx
from utils.misc import get_embedding_size


class MySimpleWrapper(nn.Module):

    def __init__(self, module, model_name, datasets=None):
        super(MySimpleWrapper, self).__init__()
        self.module = module
        self.num_features = get_embedding_size(model_name)
        # self.num_features = module.num_features
        self.model_name = model_name
        self.last_layer_name = None
        if hasattr(self.module, 'forward_features') and not ('vit' in model_name.lower()):
            self.orig_forward_features = self.module.forward_features
        else:
            # given network does not come with a forward_features
            # method like timm models
            self.orig_forward_features = None

        self.prune_output = False
        for dataset in datasets:
            if 'Dummy' in dataset:
                self.prune_output = True
                classes = get_dataset_classes('Dummy_ID')
                self.register_buffer('dummy_classes_indices', torch.LongTensor([class_to_idx[cls] for cls in classes]))


    def create_feature_extractor_sub_module(self):
        self.last_layer_name = get_graph_node_names(self.module)[0][-2]
        self.module = create_feature_extractor(self.module, [self.last_layer_name])

    def forward(self, x):
        if self.prune_output:
            return self.module.forward(x)[:, self.dummy_classes_indices]

        return self.module.forward(x)

    def forward_features(self, x):
        if not hasattr(self.module, 'forward_features'):
            return self.module(x)[self.last_layer_name]
        return self.module.forward_features(x)

    def insert_features_hook(self, hook):
        def new_forward_features(x):
            features = self.orig_forward_features(x)
            hook(features)
            return features

        self.module.forward_features = new_forward_features
