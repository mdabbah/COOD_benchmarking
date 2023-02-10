from datetime import datetime

import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.log_utils import Timer, Logger
from utils.uncertainty_metrics import gamma_correlation, ECE_calc, AURC_calc, gini
from utils.misc import to_cpu, MC_Dropout_Pass, enable_dropout


def calc_l2_dist_fast_torch(mat_a, mat_b):
    inner_products = torch.matmul(mat_a, mat_b.T)
    mat_a_inner_product = torch.matmul(mat_a, mat_a.T)
    mat_b_inner_product = torch.matmul(mat_b, mat_b.T)

    dists = torch.diag(mat_a_inner_product).unsqueeze(1) \
            + torch.diag(mat_b_inner_product).unsqueeze(0) - 2 * inner_products
    return dists


def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size


def diversity_severity_alg(severity_proxy, num_attack_groups):
    chosen_attack_groups_ids = [0]
    for _ in range(num_attack_groups):
        chosen_attack_severities = severity_proxy[chosen_attack_groups_ids]
        dist_from_already_chosen = np.min(
            np.abs(chosen_attack_severities[:, np.newaxis] - severity_proxy[np.newaxis, :]), axis=0)
        next_attacker = np.argmax(dist_from_already_chosen)
        chosen_attack_groups_ids.append(next_attacker)

    chosen_severities = severity_proxy[chosen_attack_groups_ids]
    percentiles = np.array([np.mean(chosen_severities < s) for s in chosen_severities])
    return chosen_attack_groups_ids, chosen_severities, percentiles


def get_attackers(targeted_encodings, all_attackers_encodings, num_attack_groups, distance_criterion, attack_group_size,
                  choice_alg):
    """
    :param targeted_encodings: matrix C-by-e
    row i is the encoding vector (of size e) of class i.
    :param all_attackers_encodings: matrix A-by-e
    row i is the encoding vector (of size e) of class i.
    :param num_attack_groups: number of groups to be returned x.
    :param distance_criterion: one of ( 'inner_product', 'cosine' or  'l2' ).
    :return: returns x attack groups each group is of the same size of the targeted classes
    each group is of different severity.
    the groups returned are sorted by severity (first member is most severe).
    each row is an attack group.
    """

    assert num_attack_groups < all_attackers_encodings.shape[0]
    if distance_criterion == 'inner_product':
        dists = np.matmul(targeted_encodings, all_attackers_encodings.T)
        # the bigger the value the more dangerous the attacker.
        # so to be consistent with other distance metrics we'll negate it.
        # this is still not a proper distance.
        dists = -dists
    elif distance_criterion == 'cosine':
        dists = -np.matmul(normalize(targeted_encodings, norm='l2', axis=1),
                           normalize(all_attackers_encodings.T, norm='l2', axis=0))

    elif distance_criterion == 'l2':
        mat_a, mat_b = targeted_encodings, all_attackers_encodings
        inner_products = np.matmul(mat_a, mat_b.T)
        mat_a_inner_product = np.matmul(mat_a, mat_a.T)
        mat_b_inner_product = np.matmul(mat_b, mat_b.T)

        dists = np.diag(mat_a_inner_product)[:, np.newaxis] \
                + np.diag(mat_b_inner_product)[np.newaxis, :] - 2 * inner_products
    else:
        raise ValueError("The distance criterion sent is not supported.")

    # now we have a matrix dists s.t. cell dists[i,j] says how "far" is target class i from
    # attacker class j. the more closer (or smaller the value) the attacker the more dangerous it is.

    # attackers_sorted = np.argsort(dists, axis=1)  # each column represents a different attack group.
    # dists_sorted = np.sort(dists, axis=1)  # we cannot guarantee that each attack group is of size C.
    # one attacker can be the "best" attacker to multiple classes.

    # average distance is a severity proxy
    severity_proxy = np.mean(dists, axis=0)
    attackers_ids = np.argsort(severity_proxy)
    severity_proxy = np.sort(severity_proxy)

    groups_severity = moving_average(severity_proxy, attack_group_size)

    if choice_alg == 'diversity':
        chosen_attack_groups_ids, attack_severities, percentiles = diversity_severity_alg(groups_severity,
                                                                                          num_attack_groups)
    elif choice_alg == 'fixed_percentiles':
        chosen_attack_groups_ids, attack_severities, percentiles = percentile_severity_alg(groups_severity,
                                                                                           num_attack_groups)
    else:
        raise ValueError('given choice alg is not supported')

    attack_groups = np.array([attackers_ids[idx: idx + attack_group_size] for idx in chosen_attack_groups_ids])
    return {'attack_groups': attack_groups, 'attack_severities': attack_severities, 'percentiles': percentiles}


def max_softmax_proxy(softmax_scores, labels):
    softmax_proxy = []
    for y in np.unique(labels):
        avg_softmax = np.mean(softmax_scores[labels == y])
        softmax_proxy.append(avg_softmax)

    return np.array(softmax_proxy)


def get_per_class_centroids(model, data_loader, num_classes, device, return_counts=False):
    """
    :param model: the model to extract features with. assumes features = model.forward_features(x) $\in R^e$.
    :param data_loader: the dataset loader
    :param num_classes: number of classes in the dataset.
    :param device: torch.device to do calculations on.
    :param return_counts: returns an array that holds in cell i the number of instances of class i
    in the dataset.
    :return: returns class representatives (or centroids).
    a matrix of size num_classes-by-e
    where e is the size of embedding dimension.
    """

    model.eval()
    model.float()
    with torch.no_grad():
        features = model.forward_features(data_loader.dataset[0][0].unsqueeze(0).to(f'cuda:{device}'))
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, output_size=[1, 1])

        embedding_size = int(features.shape[1])

    centroids = torch.zeros([num_classes, embedding_size]).to(f'cuda:{device}')
    class_counts = torch.zeros([num_classes]).to(f'cuda:{device}')

    z = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.float().to(f'cuda:{device}')
            y = y.to(f'cuda:{device}')

            # with Timer(f'forward pass time {device}:', print_human_readable=False):
            features = model.forward_features(x)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, output_size=[1, 1]).squeeze()

            # with Timer('loop time:', print_human_readable=False):
            # with Timer('unique_time:'):
            batch_classes = torch.unique(y)

            if len(batch_classes) == 1:
                c_ = batch_classes[0]
                centroids[c_, :] += torch.sum(features, dim=0)
                class_counts[c_] += y.shape[0]
            else:

                in_class = batch_classes[:, None] == y[None, :]

                for i, c_ in enumerate(batch_classes):  # couldn't think of a faster way that isn't a bug
                    # with Timer('one loop time:', print_human_readable=False):
                    centroids[c_, :] += torch.sum(features[in_class[i], :], dim=0)
                    class_counts[c_] += torch.sum(in_class[i])

            # z += 1
            # if z > 3:
            #     break

            # a faster way (maybe?)
            # with Timer('cache time:'):
            #     cache = torch.zeros([num_classes, features.shape[0], embedding_size]).to(f'cuda:{device}')
            #     g = {}
            #     place = [g.get(i, -1) + 1 for i in y]
            #     cache[y, place, :] = features
            #     centroids += torch.sum(cache, dim=1)
            #     class_counts += torch.sum(y[:, np.newaxis] == all_classes[np.newaxis, :], dim=0)

            # this is a bug:
            # centroids[y, :] += features
            # class_counts[y] += 1

    if return_counts:
        return to_cpu(centroids), to_cpu(class_counts)

    centroids /= class_counts.unsqueeze(0)
    return to_cpu(centroids)


def get_attackers_with_proxy(severity_proxy, num_attack_groups, attack_group_size, choice_alg):
    attackers_ids = np.argsort(severity_proxy)
    severity_proxy = np.sort(severity_proxy)

    groups_severity = moving_average(severity_proxy, attack_group_size)

    if choice_alg == 'diversity':
        chosen_attack_groups_ids, attack_severities, percentiles = diversity_severity_alg(groups_severity,
                                                                                          num_attack_groups)
    elif choice_alg == 'fixed_percentiles':
        chosen_attack_groups_ids, attack_severities, percentiles = percentile_severity_alg(groups_severity,
                                                                                           num_attack_groups)
    else:
        raise ValueError('given choice alg is not supported')

    attack_groups = np.array([attackers_ids[idx: idx + attack_group_size] for idx in chosen_attack_groups_ids])

    return {'attack_groups': attack_groups, 'attack_severities': attack_severities, 'percentiles': percentiles}


def percentile_severity_alg(groups_severity, num_groups):
    percentiles = np.linspace(0, 1, num_groups)
    chosen_attack_groups_ids = [int(q * len(groups_severity)) for q in percentiles]
    chosen_attack_groups_ids[-1] -= 1
    attack_severities = groups_severity[chosen_attack_groups_ids]

    return chosen_attack_groups_ids, attack_severities, percentiles


features_local = None


def get_dataset_statistics(model, all_data_loader, base_centroids, attacker_classes, return_averaged, device):
    """

    :param model:
    :param all_data_loader:
    :param base_centroids:
    :param attacker_classes:
    :param return_averaged:
    :param device:
    :return:
    """

    def features_hook(features_output):
        global features_local
        print(f'im in hook')
        features_local = features_output.detach().clone()

    global features_local
    model.eval()
    model.float()

    model.insert_features_hook(features_hook)
    embedding_size = model.num_features

    num_attacker_classes = len(attacker_classes)
    num_base_classes = base_centroids.shape[0]
    attackers_centroids = torch.zeros([num_attacker_classes, embedding_size]).to(f'cuda:{device}')
    attackers_class_counts = torch.zeros([num_attacker_classes]).to(f'cuda:{device}')
    attackers_avg_softmax = torch.zeros([num_attacker_classes]).to(f'cuda:{device}')

    confidences = {'softmax_conf': [], 'entropy_conf': [], 'correct': [], 'dists_conf': [], 'predictions': [], 'labels':
        []}

    i = 0
    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')
            logits = model(x)
            # print(f'features after pass: {features_local}')
            probs = F.softmax(logits, dim=1)
            softmax_conf, predictions = torch.max(probs, dim=1)
            predictions = to_cpu(predictions)
            correct = y.numpy() == predictions
            entropy_conf = Categorical(probs=probs).entropy()

            confidences['softmax_conf'].append(to_cpu(softmax_conf))
            confidences['entropy_conf'].append(to_cpu(entropy_conf))
            confidences['correct'].append(correct)
            confidences['predictions'].append(predictions)
            confidences['labels'].append(y.numpy())

            if len(features_local.shape) > 2:
                features_local = F.adaptive_avg_pool2d(features_local, output_size=[1, 1]).squeeze()

            dists_conf = torch.min(torch.cdist(features_local, base_centroids), dim=1)[0]
            confidences['dists_conf'].append(to_cpu(dists_conf))

            # with Timer('loop time:', print_human_readable=False):
            classes_to_update = np.intersect1d(np.unique(y), attacker_classes)
            for c_ in classes_to_update:  # couldn't think of a faster way that isn't a bug

                attackers_centroids[c_ - num_base_classes, :] += torch.sum(features_local[c_ == y, :], dim=0)
                attackers_class_counts[c_ - num_base_classes] += torch.sum(c_ == y)
                attackers_avg_softmax[c_ - num_base_classes] += torch.sum(softmax_conf[c_ == y])

            # i += 1
            # if i > 3:
            #     break

    if return_averaged:
        attackers_centroids /= attackers_class_counts.unsqueeze(1)
        attackers_avg_softmax /= attackers_class_counts

    return {'attackers_centroids': to_cpu(attackers_centroids),
            'attackers_class_counts': to_cpu(attackers_class_counts),
            'attackers_avg_softmax': to_cpu(attackers_avg_softmax),
            'confidences': confidences}


def calc_per_class_severity(confidence_results, confidence_field_name):
    labels = confidence_results['labels']
    attack_classes = np.unique(labels)
    num_attack_classes = len(attack_classes)
    per_instance_severity = confidence_results[confidence_field_name]
    per_class_severity_sum = np.zeros([num_attack_classes, ])
    per_class_counts = np.zeros([num_attack_classes, ])
    for i, s in enumerate(per_instance_severity):
        per_class_severity_sum[labels[i]] += s
        per_class_counts[labels[i]] += 1

    return per_class_severity_sum / per_class_counts


def extract_MC_dropout_on_dataset(model, all_data_loader, device):
    """

     :param model:
     :param all_data_loader:
     :param device:
     :return:
     """

    model.eval()
    enable_dropout(model)
    model.float()

    confidences = {'mcd_softmax_conf': [], 'entropy_conf': [], 'correct': [], 'predictions': [], 'labels': [],
                   'mcd_entropy': [], 'mutual_information': []}

    progress_log = Logger('./progress_log.txt', ['init'], False)
    batch_idx = 0
    num_batches = len(all_data_loader)
    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')

            res_dict = MC_Dropout_Pass(x, model, dropout_iterations=30, classification=True)

            predictions = to_cpu(res_dict['label_predictions'])
            correct = y.numpy() == predictions

            probs = res_dict['mean_p']
            softmax_conf, _ = torch.max(probs, dim=1)
            confidences['mcd_entropy'].append(to_cpu(res_dict['entropy_conf']))
            confidences['mutual_information'].append(to_cpu(res_dict['mutual_information']))

            entropy_conf = Categorical(probs=probs).entropy()
            confidences['mcd_softmax_conf'].append(to_cpu(softmax_conf))
            confidences['entropy_conf'].append(to_cpu(entropy_conf))
            confidences['correct'].append(correct)
            confidences['predictions'].append(predictions)
            confidences['labels'].append(y.numpy())

            progress_log.log_msg(f'{model.model_name} on device {device} finished {batch_idx}/{num_batches}  '
                                 f'time_stamp: {datetime.now()}')
            batch_idx += 1

            del x
            del probs
            del softmax_conf
            del entropy_conf
            del res_dict

            # if batch_idx == 3:
            #     break

    return confidences


def extract_softmax_on_dataset(model, data_loader, device):
    """

    :param model:
    :param data_loader:
    :param device:
    :return:
    """

    model.eval()
    model.float()

    confidences = {'confidences': [], 'correct': [], 'predictions': [], 'labels': []}

    with torch.no_grad():
        for x, y in data_loader:
            x = x.float().to(f'cuda:{device}')
            logits = model(x)

            probs = F.softmax(logits, dim=1)
            softmax_conf, predictions = torch.max(probs, dim=1)
            predictions = to_cpu(predictions)
            correct = y.numpy() == predictions

            confidences['confidences'].append(to_cpu(softmax_conf))
            confidences['correct'].append(correct)
            confidences['predictions'].append(predictions)
            confidences['labels'].append(y.numpy())

            # need to delete in explicit fashion to reduce OOM errors
            del x
            del probs
            del softmax_conf

    return confidences


def extract_entropy_on_dataset(model, all_data_loader, device):
    """

    :param model:
    :param all_data_loader:
    :param device:
    :return:
    """

    model.eval()
    model.float()

    confidences = {'confidences': [], 'correct': [], 'predictions': [], 'labels': []}

    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')
            logits = model(x)

            probs = F.softmax(logits, dim=1)
            softmax_conf, predictions = torch.max(probs, dim=1)
            predictions = to_cpu(predictions)
            correct = y.numpy() == predictions
            entropy_conf = - Categorical(probs=probs).entropy()

            confidences['confidences'].append(to_cpu(entropy_conf))
            confidences['correct'].append(correct)
            confidences['predictions'].append(predictions)
            confidences['labels'].append(y.numpy())

            del x
            del probs
            del entropy_conf

    return confidences


def extract_max_logit_on_dataset(model, all_data_loader, device):
    """

    :param model:
    :param all_data_loader:
    :param device:
    :return:
    """

    model.eval()
    model.float()

    confidences = {'confidences': [], 'correct': [], 'predictions': [], 'labels': []}

    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')
            logits = model(x)
            max_logit_conf = torch.max(logits, dim=1)[0]
            confidences['confidences'].append(to_cpu(max_logit_conf))

            probs = F.softmax(logits, dim=1)
            softmax_conf, predictions = torch.max(probs, dim=1)
            predictions = to_cpu(predictions)
            correct = y.numpy() == predictions

            confidences['correct'].append(correct)
            confidences['predictions'].append(predictions)
            confidences['labels'].append(y.numpy())

            del x
            del probs

    return confidences


def extract_odin_confidences_on_dataset(model, all_data_loader, device, confidence_args=None):
    """ implements: https://arxiv.org/pdf/1706.02690.pdf"""

    if confidence_args is None:
        temperature = 1000
        noiseMagnitude1 = 0.0014
    else:
        temperature = confidence_args['temperature']
        noiseMagnitude1 = confidence_args['noise_mag']

    model.eval()
    # model.requires_grad_(True)
    # for param in model.module.parameters():
    #     param.requires_grad = True
    model.float()

    confidences = {'softmax_conf': [], 'entropy_conf': [], 'correct': [], 'predictions': [], 'labels':
        [], 'odin_conf': []}

    temperature = torch.tensor(temperature).float().to(f'cuda:{device}')
    noiseMagnitude1 = torch.tensor(noiseMagnitude1).float().to(f'cuda:{device}')

    # image_norm_vector = torch.tensor([0.485, 0.456, 0.406]).float().to(f'cuda:{device}')
    # IDO and GUY check this vector I asume it's supposed to be the std vec for normalizing images
    # I took it from torch ...
    image_norm_vector = torch.tensor([0.229, 0.224, 0.225]).float().to(f'cuda:{device}')
    # odin github (https://github.com/facebookresearch/odin/blob/main/code/calData.py)
    # used this vector [(63.0 / 255.0), (62.1 / 255.0), (66.7 / 255.0)]
    # image_norm_vector = torch.tensor([(63.0 / 255.0), (62.1 / 255.0), (66.7 / 255.0)]).float().to(f'cuda:{device}')

    image_norm_vector = image_norm_vector.view((1, 3, 1, 1))

    opt = torch.optim.SGD([*model.parameters(), image_norm_vector, temperature, noiseMagnitude1], lr=0.001)

    # with torch.no_grad():
    for x, y in all_data_loader:
        x = x.float().to(f'cuda:{device}')
        x.requires_grad_(True)
        opt.zero_grad()

        logits = model(x)

        probs = F.softmax(logits, dim=1)
        softmax_conf, predictions = torch.max(probs, dim=1)
        loss = F.cross_entropy(probs, predictions)

        predictions = to_cpu(predictions)
        correct = y.numpy() == predictions
        entropy_conf = - Categorical(probs=probs).entropy()

        loss.backward()

        # sign(gradient_x)
        gradient = torch.ge(x.grad, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the same space of image
        gradient = gradient / image_norm_vector

        # Adding small perturbations to images
        perturbed_x = x - (noiseMagnitude1 * gradient)

        # forward pass the perturbed image
        new_logits = model(perturbed_x)
        new_logits = new_logits / temperature

        # for numerical stability ...
        new_logits = new_logits - (torch.max(new_logits, dim=1)[0]).view(len(new_logits), 1)
        # Calculating the confidence after adding perturbations
        new_probs = F.softmax(new_logits, dim=1)
        odin_conf = torch.max(new_probs, dim=1)[0]

        confidences['softmax_conf'].append(to_cpu(softmax_conf))
        confidences['odin_conf'].append(to_cpu(odin_conf))
        confidences['entropy_conf'].append(to_cpu(entropy_conf))
        confidences['correct'].append(correct)
        confidences['predictions'].append(predictions)
        confidences['labels'].append(y)

        del x
        del y
        del probs
        del softmax_conf
        del entropy_conf
        del odin_conf
        del new_probs
        del new_logits
        del perturbed_x
        del gradient
        del loss

        torch.cuda.empty_cache()

    return confidences


def extract_softmax_signals_on_dataset(model, all_data_loader, device):
    """

    :param model:
    :param all_data_loader:
    :param device:
    :return:
    """

    model.eval()
    model.float()

    confidences = {'softmax_conf': [], 'entropy_conf': [], 'correct': [], 'predictions': [], 'labels':
        [], 'max_logit_conf': []}

    progress_log = Logger('./progress_log.txt', ['init'], False)
    batch_idx = 0
    num_batches = len(all_data_loader)
    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')
            logits = model(x)

            probs = F.softmax(logits, dim=1)
            softmax_conf, predictions = torch.max(probs, dim=1)
            predictions = to_cpu(predictions)
            correct = y.numpy() == predictions
            entropy_conf = Categorical(probs=probs).entropy()

            confidences['softmax_conf'].append(to_cpu(softmax_conf))
            confidences['entropy_conf'].append(to_cpu(entropy_conf))
            confidences['correct'].append(correct)
            confidences['predictions'].append(predictions)
            confidences['labels'].append(y.numpy())

            max_logit_conf = torch.max(logits, dim=1)[0]
            confidences['max_logit_conf'].append(to_cpu(max_logit_conf))

            progress_log.log_msg(f'{model.model_name} on device {device} finished {batch_idx}/{num_batches}  '
                                 f'time_stamp: {datetime.now()}')
            batch_idx += 1

            del x
            del probs
            del softmax_conf
            del entropy_conf

            # if batch_idx == 3:
            #     break

    return confidences


def get_dataset_last_activations(model, all_data_loader, device):
    """
    for ensamble methods and temp scaling
    :param model:
    :param all_data_loader:
    :param device:
    :return:
    """

    model.eval()
    model.float()

    confidences = {'logits': [], 'softmax': [], 'labels': []}

    progress_log = Logger('./progress_log.txt', ['init'], False)
    batch_idx = 0
    num_batches = len(all_data_loader)
    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')
            logits = model(x)

            probs = F.softmax(logits, dim=1)

            confidences['logits'].append(to_cpu(logits))
            confidences['softmax'].append(to_cpu(probs))
            confidences['labels'].append(y.numpy())

            progress_log.log_msg(f'{model.model_name} on device {device} finished {batch_idx}/{num_batches}  '
                                 f'time_stamp: {datetime.now()}')
            batch_idx += 1

            del x
            del probs

            # if batch_idx == 3:
            #     break

    return confidences


def get_dataset_embeddings(model, all_data_loader, device):
    """
    for density based kappa
    warning: causes model to lose classification head
    assumes model is wrapped with misc.model_calss_wrapper
    :param model:
    :param all_data_loader:
    :param device:
    :return:
    """

    model.eval()
    model.float()

    dataset_fl = {'features': [], 'labels': []}

    progress_log = Logger('./progress_log.txt', ['init'], False)
    batch_idx = 0
    num_batches = len(all_data_loader)
    if 'clip' not in model.model_name.lower():
        model.create_feature_extractor_sub_module()

    with torch.no_grad():
        for x, y in all_data_loader:
            x = x.float().to(f'cuda:{device}')
            features = model.forward_features(x)

            if isinstance(features, tuple):
                features = features[0]

            if len(features.shape) > 3:
                features = F.adaptive_avg_pool2d(features, output_size=[1, 1]).squeeze()

            dataset_fl['labels'].append(y.numpy())
            dataset_fl['features'].append(to_cpu(features))

            progress_log.log_msg(f'{model.model_name} on device {device} finished {batch_idx}/{num_batches}  '
                                 f'time_stamp: {datetime.now()}')
            batch_idx += 1

            del x
            del features

            # if batch_idx == 3:
            #     break

    return {k: np.concatenate(v) for k, v in dataset_fl.items()}


def calc_per_class_gaussian_parameters(features, labels):
    """
    given a 2d matrix features of samples (each row is a sample)
    and the labels of those samples.
    we estimate the parameters of a gaussian for each class:
    mu, sigma
    :param features:  2d matrix features of samples (each row is a sample) [num sample x num features]
    :param labels: labels of those samples.
    :return: dict['mu', 'sigma', 'classes', 'class_cunts'].
    """

    classes, class_cunts = np.unique(labels, return_counts=True)
    num_classes = class_cunts.shape[0]
    embedding_size = features.shape[1]
    centroids = np.zeros([num_classes, embedding_size])
    sigmas = np.zeros([num_classes, embedding_size, embedding_size])
    precisions = np.zeros([num_classes, embedding_size, embedding_size])

    for class_label in classes:
        class_samples = labels == class_label
        class_features = features[class_samples]
        n = class_features.shape[0]

        # estimate mu
        centroids[class_label] = np.mean(class_features, axis=0)

        # center features

        import sklearn.covariance
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(class_features)
        precisions[class_label] = group_lasso.precision_

        class_features -= centroids[class_label]

        outer_products = np.einsum('ij, ik-> ijk', class_features, class_features)

        sigmas[class_label] = np.sum(outer_products, axis=0) / (n - 1)  # unbiased estimate

    return {'mu': centroids, 'sigmas': sigmas, 'classes': classes, 'class_cunts': class_cunts,
            'precisions': precisions}


def fix_sigmas(sigmas, noise_mag=0.0001):
    recombined_sigmas = []
    for c_ in range(sigmas.shape[0]):
        u, s, v = np.linalg.svd(sigmas[c_], compute_uv=True)
        s[s <= 0] = np.max([noise_mag, np.min(s)])

        recombined = (u * s) @ v
        recombined_sigmas.append(recombined)

    return np.stack(recombined_sigmas, axis=0)


def calc_density_based_confidences(gaussians, samples_features, fix_type='ridge', noise_mag=0):
    if 'sigmas_inv' not in gaussians:
        with Timer(f'time to fix sigmas with {fix_type}'):
            sigmas = gaussians['sigmas']
            if fix_type == 'ridge':
                gaussians['sigmas_inv'] = np.linalg.inv(sigmas + noise_mag * np.eye(sigmas.shape[-1]))
            elif fix_type == 'svd':
                gaussians['sigmas_inv'] = np.linalg.inv(fix_sigmas(sigmas, noise_mag))
    centroids = gaussians['mu']
    sigmas_inv = gaussians['sigmas_inv']

    class_cunts = gaussians['class_cunts']
    class_priors = class_cunts / np.sum(class_cunts)

    confidences = {'mahalanobis_conf': [], 'ddu_posterior': [], 'ddu_uniform': [], 'ddu_exponent': [],
                   'ddu_exponent_posterior': []}

    k = samples_features.shape[1]  # features_dim
    # norm_const = np.power(2 * np.pi, -k/2)
    two_pow = np.power(2, k / 2)
    pi_pow = np.power(np.pi, k / 2)
    # per_class_norm = np.sqrt(np.linalg.det(sigmas_inv))/two_pow  # 1/det(sigma) == det(sigma_inverted)
    per_class_norm = np.linalg.det(sigmas_inv)
    per_class_norm[per_class_norm < 0] = 0
    per_class_norm = np.sqrt(per_class_norm)
    per_class_norm /= pi_pow

    num_samples = samples_features.shape[0]
    batch_size = 128
    for batch_start_idx in range(0, num_samples, batch_size):
        batch_features = samples_features[batch_start_idx: batch_start_idx + batch_size]

        if len(batch_features) == 0:
            break
        batch_features = batch_features[np.newaxis, :, :] - centroids[:, np.newaxis, :]

        mahalanobis_dist_sq = np.einsum('cif,cfk,cik->ci', batch_features, sigmas_inv, batch_features)
        mahalanobis_dist_sq[mahalanobis_dist_sq < 0] = 0
        mahalanobis_dist = np.sqrt(mahalanobis_dist_sq)

        confidences['mahalanobis_conf'].append(np.max(-mahalanobis_dist, axis=0))

        ddu_exp = np.exp(-1 / 2 * mahalanobis_dist_sq)

        ddu_uniform = np.einsum('c,ci-> ci', per_class_norm, ddu_exp)
        ddu_post = np.einsum('c,ci-> ci', class_priors, ddu_uniform)

        confidences['ddu_exponent'].append(np.sum(ddu_exp, axis=0))

        ddu_uniform = np.sum(ddu_uniform, axis=0)
        ddu_posterior = np.sum(ddu_post, axis=0)

        confidences['ddu_uniform'].append(ddu_uniform)
        confidences['ddu_posterior'].append(ddu_posterior)

        ddu_exp_post = np.einsum('c,ci-> ci', class_priors, ddu_exp)
        ddu_exp_post = np.sum(ddu_exp_post, axis=0)
        confidences['ddu_exponent_posterior'].append(ddu_exp_post)

    # stack and return
    return {k: np.concatenate(v) for k, v in confidences.items()}


def calc_density_based_confidences_accelerated(gaussians, samples_features, fix_type='ridge', noise_mag=0,
                                               batch_size=128):
    if 'sigmas_inv' not in gaussians:
        with Timer(f'time to fix sigmas with {fix_type}'):
            sigmas = gaussians['sigmas']
            if fix_type == 'ridge':
                gaussians['sigmas_inv'] = np.linalg.inv(sigmas + noise_mag * np.eye(sigmas.shape[-1]))
            elif fix_type == 'svd':
                gaussians['sigmas_inv'] = np.linalg.inv(fix_sigmas(sigmas, noise_mag))

    centroids = torch.from_numpy(gaussians['mu']).cuda()
    sigmas_inv = torch.from_numpy(gaussians['sigmas_inv']).cuda()

    class_cunts = gaussians['class_cunts']
    class_priors = torch.from_numpy(class_cunts / np.sum(class_cunts)).cuda()

    confidences = {'mahalanobis_conf': [], 'ddu_posterior': [], 'ddu_uniform': [], 'ddu_exponent': [],
                   'ddu_exponent_posterior': []}

    k = samples_features.shape[1]  # features_dim
    norm_const = np.power(2 * np.pi, -k / 2)
    torch_pi = torch.tensor(np.pi).cuda()
    torch_two = torch.tensor(2).cuda()

    per_class_norm = torch.linalg.det(sigmas_inv)
    per_class_norm[per_class_norm < 0] = 0
    per_class_norm = torch.sqrt(per_class_norm)

    # per_class_norm = torch.pow(2*torch_pi, -k/2) * torch.sqrt(torch.linalg.det(sigmas_inv))
    per_class_norm /= torch_pi.pow(k / 2)
    per_class_norm /= torch_two.pow(k / 2)

    num_samples = samples_features.shape[0]
    # batch_size = 128
    for batch_start_idx in range(0, num_samples, batch_size):
        batch_features = samples_features[batch_start_idx: batch_start_idx + batch_size]
        if len(batch_features) == 0:
            break
        batch_features = torch.from_numpy(batch_features).cuda()
        batch_features = batch_features[np.newaxis, :, :] - centroids[:, np.newaxis, :]

        mahalanobis_dist_sq = torch.einsum('cif,cfk,cik->ci', batch_features, sigmas_inv, batch_features)
        mahalanobis_dist_sq[mahalanobis_dist_sq < 0] = 0
        mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq)

        mahalanobis_conf = to_cpu(torch.max(-mahalanobis_dist, dim=0)[0])
        confidences['mahalanobis_conf'].append(mahalanobis_conf)

        ddu_exp = torch.exp(-1 / 2 * mahalanobis_dist_sq)

        ddu_uniform = torch.einsum('c,ci-> ci', per_class_norm, ddu_exp)
        ddu_post = torch.einsum('c,ci-> ci', class_priors, ddu_uniform)

        confidences['ddu_exponent'].append(to_cpu(torch.sum(ddu_exp, dim=0)))

        ddu_uniform = to_cpu(torch.sum(ddu_uniform, dim=0))
        ddu_posterior = to_cpu(torch.sum(ddu_post, dim=0))

        confidences['ddu_uniform'].append(ddu_uniform)
        confidences['ddu_posterior'].append(ddu_posterior)

        ddu_exp_post = torch.einsum('c,ci-> ci', class_priors, ddu_exp)
        ddu_exp_post = torch.sum(ddu_exp_post, dim=0)
        confidences['ddu_exponent_posterior'].append(to_cpu(ddu_exp_post))

    # stack and return
    return {k: np.concatenate(v) for k, v in confidences.items()}


def calc_density_based_confidences_accelerated_v2(gaussians, samples_features, fix_type='ridge', noise_mag=0,
                                                  batch_size=128):
    if 'sigmas_inv' not in gaussians:
        with Timer(f'time to fix sigmas with {fix_type}'):
            sigmas = gaussians['sigmas']
            if fix_type == 'ridge':
                gaussians['sigmas_inv'] = np.linalg.inv(sigmas + noise_mag * np.eye(sigmas.shape[-1]))
            elif fix_type == 'svd':
                gaussians['sigmas_inv'] = np.linalg.inv(fix_sigmas(sigmas, noise_mag))

    c_batch_size = 250
    num_classes = 1000
    confidences = {'mahalanobis_conf': [], 'ddu_v2': []}
    for c_batch_start_idx in range(0, num_classes, c_batch_size):
        centroids = torch.from_numpy(gaussians['mu'][c_batch_start_idx: c_batch_start_idx + c_batch_size]).cuda()
        sigmas_inv = torch.from_numpy(
            gaussians['sigmas_inv'][c_batch_start_idx: c_batch_start_idx + c_batch_size]).cuda()

        class_cunts = gaussians['class_cunts']
        class_priors = torch.from_numpy(class_cunts / np.sum(class_cunts)).cuda()
        class_priors = class_priors[c_batch_start_idx: c_batch_start_idx + c_batch_size]

        confidences_cbatch = {'mahalanobis_conf': [], 'ddu_v2': []}

        # ddu_v2 is \sum_c exp(log( p_c(x)))
        # log( p_c(x)) = -1/2[k*log(2pi) + log(SIGMA_c) + Mahalanobis(x, mu_c, SIGMA_c)^2]

        # log( p_c(x)) =  1/2log(SIGMA_inv_c) + -1/2[k*log(2pi)  + Mahalanobis(x, mu_c, SIGMA_c)^2]

        k = samples_features.shape[1]  # features_dim
        norm_const = np.power(2 * np.pi, -k / 2)
        torch_pi = torch.tensor(np.pi).cuda()
        torch_two = torch.tensor(2).cuda()

        klog2pi = k * torch.tensor(np.log(2 * np.pi)).cuda()

        log_sigma_inv_det = torch.linalg.det(sigmas_inv)
        log_sigma_inv_det[log_sigma_inv_det < 0] = 0
        per_class_norm = torch.sqrt(log_sigma_inv_det)

        log_sigma_inv_det = torch.log(log_sigma_inv_det)

        # per_class_norm = torch.pow(2*torch_pi, -k/2) * torch.sqrt(torch.linalg.det(sigmas_inv))
        per_class_norm /= torch_pi.pow(k / 2)
        per_class_norm /= torch_two.pow(k / 2)

        num_samples = samples_features.shape[0]
        # batch_size = 128
        for batch_start_idx in range(0, num_samples, batch_size):
            batch_features = samples_features[batch_start_idx: batch_start_idx + batch_size]
            if len(batch_features) == 0:
                break
            batch_features = torch.from_numpy(batch_features).cuda()
            batch_features = batch_features[np.newaxis, :, :] - centroids[:, np.newaxis, :]

            mahalanobis_dist_sq = torch.einsum('cif,cfk,cik->ci', batch_features, sigmas_inv, batch_features)
            mahalanobis_dist_sq[mahalanobis_dist_sq < 0] = 0
            mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq)

            mahalanobis_conf = to_cpu(torch.max(-mahalanobis_dist, dim=0)[0])
            confidences_cbatch['mahalanobis_conf'].append(mahalanobis_conf)

            # print(f'shapes are {log_sigma_inv_det.shape}  {klog2pi.shape}  {mahalanobis_dist_sq.shape}')
            ddu_v2 = torch.exp(
                0.5 * (log_sigma_inv_det.unsqueeze(1) - klog2pi.unsqueeze(0).unsqueeze(0) - mahalanobis_dist_sq))
            ddu_v2 = torch.sum(ddu_v2, dim=0)

            # ddu_exp = torch.exp(-1 / 2 * mahalanobis_dist_sq)

            # ddu_uniform = torch.einsum('c,ci-> ci', per_class_norm, ddu_exp)
            # ddu_post = torch.einsum('c,ci-> ci', class_priors, ddu_uniform)

            # confidences_cbatch['ddu_exponent'].append(to_cpu(torch.sum(ddu_exp, dim=0)))

            # ddu_uniform = to_cpu(torch.sum(ddu_uniform, dim=0))
            # ddu_posterior = to_cpu(torch.sum(ddu_post, dim=0))

            # confidences_cbatch['ddu_uniform'].append(ddu_uniform)
            # confidences_cbatch['ddu_posterior'].append(ddu_posterior)

            # ddu_exp_post = torch.einsum('c,ci-> ci', class_priors, ddu_exp)
            # ddu_exp_post = torch.sum(ddu_exp_post, dim=0)
            # confidences_cbatch['ddu_exponent_posterior'].append(to_cpu(ddu_exp_post))

            confidences_cbatch['ddu_v2'].append(to_cpu(ddu_v2))  # shape: #instances x #classes

            # if batch_start_idx > 3*batch_size:
            #     break

        # stack and return
        confidences_cbatch = {k: np.concatenate(v) for k, v in confidences_cbatch.items()}
        for k, v in confidences_cbatch.items():
            confidences[k].append(v)

    # for k, v in confidences.items():
    #     confidences = np.concatenate(confidences[k], axis=1)
    confidences = {k: np.stack(v, axis=1) for k, v in confidences.items()}
    confidences['mahalanobis_conf'] = np.max(confidences['mahalanobis_conf'], axis=1)
    # confidences['ddu_posterior'] = np.sum(confidences['ddu_posterior'], axis=1)
    # confidences['ddu_uniform'] = np.sum(confidences['ddu_uniform'], axis=1)
    # confidences['ddu_exponent'] = np.sum(confidences['ddu_exponent'], axis=1)
    # confidences['ddu_exponent_posterior'] = np.sum(confidences['ddu_exponent_posterior'], axis=1)
    confidences['ddu_v2'] = np.sum(confidences['ddu_v2'], axis=1)

    return confidences


def calc_mahalanobis_confidence(gaussians, features_and_labels, batch_size=128):
    c_batch_size = 250
    num_classes = 1000
    samples_features = features_and_labels['features']
    labels = features_and_labels['labels']
    confidences = {'mahalanobis_conf': [], 'ddu_v2': []}
    for c_batch_start_idx in range(0, num_classes, c_batch_size):
        centroids = torch.from_numpy(gaussians['mu'][c_batch_start_idx: c_batch_start_idx + c_batch_size]).cuda()
        sigmas_inv = torch.from_numpy(
            gaussians['precisions'][c_batch_start_idx: c_batch_start_idx + c_batch_size]).cuda()

        class_cunts = gaussians['class_cunts']
        class_priors = torch.from_numpy(class_cunts / np.sum(class_cunts)).cuda()
        class_priors = class_priors[c_batch_start_idx: c_batch_start_idx + c_batch_size]

        confidences_cbatch = {'mahalanobis_conf': [], 'ddu_v2': []}

        # ddu_v2 is \sum_c exp(log( p_c(x))) where
        # log( p_c(x)) = -1/2[k*log(2pi) + log(SIGMA_c) + Mahalanobis(x, mu_c, SIGMA_c)^2]

        # log( p_c(x)) =  1/2log(SIGMA_inv_c) + -1/2[k*log(2pi)  + Mahalanobis(x, mu_c, SIGMA_c)^2]

        k = samples_features.shape[1]  # features_dim
        norm_const = np.power(2 * np.pi, -k / 2)
        torch_pi = torch.tensor(np.pi).cuda()
        torch_two = torch.tensor(2).cuda()

        klog2pi = k * torch.tensor(np.log(2 * np.pi)).cuda()

        log_sigma_inv_det = torch.linalg.det(sigmas_inv)
        log_sigma_inv_det[log_sigma_inv_det < 0] = 0
        per_class_norm = torch.sqrt(log_sigma_inv_det)

        log_sigma_inv_det = torch.log(log_sigma_inv_det)

        # per_class_norm = torch.pow(2*torch_pi, -k/2) * torch.sqrt(torch.linalg.det(sigmas_inv))
        per_class_norm /= torch_pi.pow(k / 2)
        per_class_norm /= torch_two.pow(k / 2)

        num_samples = samples_features.shape[0]
        # batch_size = 128
        for batch_start_idx in range(0, num_samples, batch_size):
            batch_features = samples_features[batch_start_idx: batch_start_idx + batch_size]
            if len(batch_features) == 0:
                break
            batch_features = torch.from_numpy(batch_features).cuda()
            batch_features = batch_features[np.newaxis, :, :] - centroids[:, np.newaxis, :]

            mahalanobis_dist_sq = torch.einsum('cif,cfk,cik->ci', batch_features, sigmas_inv, batch_features)
            mahalanobis_dist_sq[mahalanobis_dist_sq < 0] = 0
            mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq)

            mahalanobis_conf = to_cpu(torch.max(-mahalanobis_dist, dim=0)[0])
            confidences_cbatch['mahalanobis_conf'].append(mahalanobis_conf)

            # print(f'shapes are {log_sigma_inv_det.shape}  {klog2pi.shape}  {mahalanobis_dist_sq.shape}')
            ddu_v2 = torch.exp(
                0.5 * (log_sigma_inv_det.unsqueeze(1) - klog2pi.unsqueeze(0).unsqueeze(0) - mahalanobis_dist_sq))
            ddu_v2 = torch.sum(ddu_v2, dim=0)

            confidences_cbatch['ddu_v2'].append(to_cpu(ddu_v2))  # shape: #instances x #classes

            # if batch_start_idx > 3*batch_size:
            #     break

        # stack and return
        confidences_cbatch = {k: np.concatenate(v) for k, v in confidences_cbatch.items()}
        for k, v in confidences_cbatch.items():
            confidences[k].append(v)

    # for k, v in confidences.items():
    #     confidences = np.concatenate(confidences[k], axis=1)
    confidences = {k: np.stack(v, axis=1) for k, v in confidences.items()}
    confidences['mahalanobis_conf'] = np.max(confidences['mahalanobis_conf'], axis=1)

    confidences['ddu_v2'] = np.sum(confidences['ddu_v2'], axis=1)
    confidences['labels'] = features_and_labels['labels']
    return confidences


def test_l2_fastest(num_samples_a, num_samples_b):
    a = torch.rand([num_samples_a, 2048]).cuda(0)  # simulating a batch
    b = torch.rand([num_samples_b, 2048]).cuda(0)  # simulating the base class centroids

    # with Timer('double loop time:', print_human_readable=False): #more than 10000 times slower than the other methods
    #     gt = torch.zeros([a.shape[0], b.shape[0]]).cuda(0)
    #     for i in range(a.shape[0]):
    #         for j in range(b.shape[0]):
    #             gt[i, j] = torch.sum((a[i] - b[j]) ** 2)

    # with Timer('broadcast thing:', print_human_readable=False):
    #     # gets fast the second time , don't know why
    #     # runs out of memory fast
    #     broadcast_thing = torch.sum((a.unsqueeze(1) - b.unsqueeze(0))**2, dim=-1)

    with Timer('inner product thing:', print_human_readable=False):
        inner_product_thing = calc_l2_dist_fast_torch(a, b)

    with Timer('torch cdist:', print_human_readable=False):
        # 0.1 milliseconds fsater than the inner product thing
        cdist_thing = torch.cdist(a, b) ** 2

    return cdist_thing, inner_product_thing


def calc_OOD_metrics_on_dataset(confidences, confidence_type):
    """
    :param confidences: a dictionary that contains at least the following keys:
    confidence_type : an array of the size of the number of sampels and cell i contains
    the confidence given to sample i.
    'labels': an array of the size of the number of sampels and cell i contains the label for sample i.
    'is_ID': an array of the size of the number of sampels and cell i==1 iff the sample is in distribution.
    :return: the OOD performance of the given confidence type
    for each one of the attack groups.
    pass
    """

    id_sampels = confidences['is_ID']
    confs = confidences[confidence_type]
    is_correct = confidences['correct']
    labels = confidences['labels']

    id_confs = confs[id_sampels]
    id_is_correct = is_correct[id_sampels]
    id_acc = np.mean(id_is_correct)
    id_avg_conf = np.mean(id_confs)
    id_median_conf = np.median(id_confs)
    id_std = np.std(id_confs, ddof=1)
    id_gini = gini(torch.tensor(id_confs[:, np.newaxis]))

    sample_certainties = torch.from_numpy(np.stack([id_confs, id_is_correct], axis=1))

    id_gamma_results = gamma_correlation(sample_certainties, sort=True)
    id_aurc = AURC_calc(sample_certainties, sort=True)

    assert id_confs.shape[0] == 50_000 or id_confs.shape[0] == 45_000
    attack_groups_results = []

    # attack_group is a vector of class ids that belong to the groups
    ood_samples = np.logical_not(id_sampels)

    ood_conf = confs[ood_samples]
    ood_avg_conf = np.mean(ood_conf)
    ood_median_conf = np.median(ood_conf)
    ood_std = np.std(ood_conf, ddof=1)
    attack_gini = gini(torch.tensor(ood_conf[:, np.newaxis]))

    ood_is_correct = is_correct[ood_samples]
    assert np.sum(ood_is_correct) == 0
    # assert ood_conf.shape[0] == 50_000 or ood_conf.shape[0] == 45_000

    certainties = np.concatenate([ood_conf, id_confs])
    is_id = np.concatenate([np.zeros_like(ood_conf), np.ones_like(id_confs)])

    assert len(is_id) == len(certainties)

    sample_certainties = torch.from_numpy(np.stack([certainties, is_id], axis=1))
    idx = torch.randperm(sample_certainties.shape[0])
    sample_certainties = sample_certainties[idx]
    gamma_results = gamma_correlation(sample_certainties, sort=True)
    aurc = AURC_calc(sample_certainties, sort=True)

    samples_is_correct = np.concatenate([np.zeros_like(ood_conf), id_is_correct])
    sample_certainties = torch.from_numpy(np.stack([certainties, samples_is_correct], axis=1))
    idx = torch.randperm(sample_certainties.shape[0])
    sample_certainties = sample_certainties[idx]
    correctness_gamma_results = gamma_correlation(sample_certainties, sort=True)
    correctness_aurc = AURC_calc(sample_certainties, sort=True)

    ece, mce = ECE_calc(torch.from_numpy(np.stack([ood_conf, np.zeros_like(ood_conf)], axis=1)))

    results = {'ood_avg_conf': ood_avg_conf, 'ood_median_conf': ood_median_conf,
               'ood_std': ood_std, 'ood_gini': attack_gini,
               'id_avg_conf': id_avg_conf, 'id_median_conf': id_median_conf,
               'id_std': id_std, 'id_gini': id_gini, 'id_acc': id_acc,
               'id-gamma': id_gamma_results['gamma'],
               'id-auroc': id_gamma_results['AUROC'],
               'is-aurc': id_aurc,
               'ood-gamma': gamma_results['gamma'], 'ood-auroc': gamma_results['AUROC'], 'ood-aurc': aurc,
               'correctness-gamma': correctness_gamma_results['gamma'],
               'correctness-auroc': correctness_gamma_results['AUROC'],
               'correctness-aurc': correctness_aurc, 'ece': ece, 'mce': mce,
               }

    attack_groups_results.append(results)

    return attack_groups_results


def calc_OOD_metrics(attack_groups, confidences, confidence_key):
    """
    given attack_groups where each row is a group
    :param attack_groups:
    :param confidences: a dictionary that contains at least the following keys:
    confidence_type : an array of the size of the number of sampels and cell i contains
    the confidence given to sample i.
    'labels': an array of the size of the number of sampels and cell i contains the label for sample i.
    'is_ID': an array of the size of the number of sampels and cell i==1 iff the sample is in distribution.
    :return: the OOD performance of the given confidence type
    for each one of the attack groups.
    pass
    """

    id_sampels = confidences['is_ID']
    confs = confidences[confidence_key]
    is_correct = confidences['correct']
    labels = confidences['labels']

    id_confs = confs[id_sampels]
    id_is_correct = is_correct[id_sampels]
    id_acc = np.mean(id_is_correct)
    id_avg_conf = np.mean(id_confs)
    id_median_conf = np.median(id_confs)
    id_std = np.std(id_confs, ddof=1)
    id_gini = gini(torch.tensor(id_confs[:, np.newaxis]))

    sample_certainties = torch.from_numpy(np.stack([id_confs, id_is_correct], axis=1))

    id_gamma_results = gamma_correlation(sample_certainties, sort=True)
    id_aurc = AURC_calc(sample_certainties, sort=True)

    assert id_confs.shape[0] == 50_000 or id_confs.shape[0] == 45_000
    attack_groups_results = []

    for attack_group in attack_groups:
        # attack_group is a vector of class ids that belong to the groups
        attack_samples = np.any(attack_group[:, np.newaxis] == labels[np.newaxis, :], axis=0)

        attack_conf = confs[attack_samples]
        attack_avg_conf = np.mean(attack_conf)
        attack_median_conf = np.median(attack_conf)
        attack_std = np.std(attack_conf, ddof=1)
        attack_gini = gini(torch.tensor(attack_conf[:, np.newaxis]))

        attack_is_correct = is_correct[attack_samples]
        assert np.sum(attack_is_correct) == 0
        assert attack_conf.shape[0] == 50_000 or attack_conf.shape[0] == 45_000

        certainties = np.concatenate([attack_conf, id_confs])
        is_id = np.concatenate([np.zeros_like(attack_conf), np.ones_like(id_confs)])

        assert len(is_id) == len(certainties)

        sample_certainties = torch.from_numpy(np.stack([certainties, is_id], axis=1))
        idx = torch.randperm(sample_certainties.shape[0])
        sample_certainties = sample_certainties[idx]
        gamma_results = gamma_correlation(sample_certainties, sort=True)
        aurc = AURC_calc(sample_certainties, sort=True)

        samples_is_correct = np.concatenate([np.zeros_like(attack_conf), id_is_correct])
        sample_certainties = torch.from_numpy(np.stack([certainties, samples_is_correct], axis=1))
        idx = torch.randperm(sample_certainties.shape[0])
        sample_certainties = sample_certainties[idx]
        correctness_gamma_results = gamma_correlation(sample_certainties, sort=True)
        correctness_aurc = AURC_calc(sample_certainties, sort=True)

        ece, mce = ECE_calc(torch.from_numpy(np.stack([attack_conf, np.zeros_like(attack_conf)], axis=1)))

        results = {'ood_avg_conf': attack_avg_conf, 'ood_median_conf': attack_median_conf,
                   'ood_std': attack_std, 'ood_gini': attack_gini,
                   'id_avg_conf': id_avg_conf, 'id_median_conf': id_median_conf,
                   'id_std': id_std, 'id_gini': id_gini, 'id_acc': id_acc,
                   'id-gamma': id_gamma_results['gamma'],
                   'id-auroc': id_gamma_results['AUROC'],
                   'is-aurc': id_aurc,
                   'ood-gamma': gamma_results['gamma'], 'ood-auroc': gamma_results['AUROC'], 'ood-aurc': aurc,
                   'correctness-gamma': correctness_gamma_results['gamma'],
                   'correctness-auroc': correctness_gamma_results['AUROC'],
                   'correctness-aurc': correctness_aurc, 'ece': ece, 'mce': mce,
                   }

        attack_groups_results.append(results)

    return attack_groups_results


def test_fastest_density_based():
    dim_f = 1024
    gaussians = {'sigmas': np.random.rand(1000, dim_f, dim_f), 'class_cunts': np.array([1300] * 1000),
                 'mu': np.random.rand(1000, dim_f)}

    features = np.random.rand(3000, dim_f)
    print('generated data')
    print('doing numpy')
    # with Timer('numpy time is:'):
    #     _ = calc_density_based_confidences(gaussians, features)

    print('doing torch')
    with Timer('pytorch time is:'):
        res = calc_density_based_confidences_accelerated_v2(gaussians, features)

    return res


if __name__ == '__main__':
    # model = timm.create_model('resnet50', pretrained=True)
    # transforms = get_base_transforms(224)
    # data_loader = create_data_loader('ImageNet_20K', None, batch_size=128, num_workers=8, transform=transforms)
    # num_classes = data_loader.dataset.num_classes
    # get_per_class_centroids(model, data_loader, num_classes, device='cuda')

    # test_l2_fastest(2000, 1000)
    a = test_fastest_density_based()
