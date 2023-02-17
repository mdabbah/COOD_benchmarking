import numpy as np

import torch
import numpy as np

'''
NOTE: All following methods get samples_certainties, which is a tensor of size (N,2) of N samples, for each 
its certainty and whether or not it was a correct prediction. Position 0 in each sample is its confidence, and 
position 1 is its correctness (True \ 1 for correct samples and False \ 0 for incorrect ones).

If samples_certainties is sorted in a descending order, set sort=False to avoid sorting it again.

Example: samples_certainties[0][0] is the confidence score of the first sample.
samples_certainties[0][1] is the correctness (True \ False) of the first sample.
'''


def confidence_variance(samples_certainties):
    return torch.var(samples_certainties.transpose(0, 1)[0], unbiased=True).item()


def confidence_mean(samples_certainties):
    return torch.mean(samples_certainties.transpose(0, 1)[0]).item()


def confidence_median(samples_certainties):
    return torch.median(samples_certainties.transpose(0, 1)[0]).item()


def gini(samples_certainties):
    array = samples_certainties.transpose(0, 1)[0]
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    if torch.amin(array) < 0:
        # Values cannot be negative:
        array -= torch.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = torch.sort(array)[0]
    # Index per array element:
    index = torch.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((torch.sum((2 * index - n - 1) * array)) / (n * torch.sum(array))).item()


def gamma_correlation(samples_certainties, sort=True):
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    total_samples = samples_certainties.shape[0]
    incorrect_after_me = np.zeros((total_samples))

    for i in range(total_samples - 1, -1, -1):
        if i == total_samples - 1:
            incorrect_after_me[i] = 0
        else:
            incorrect_after_me[i] = incorrect_after_me[i + 1] + (1 - int(samples_certainties[i + 1][1]))
            # Note: samples_certainties[i+1][1] is the correctness label for sample i+1
            tst = samples_certainties[i + 1][1]

    n_d = 0  # amount of different pairs of ordering
    n_s = 0  # amount of pairs with same ordering
    incorrect_before_me = 0
    for i in range(total_samples):
        if i != 0:
            incorrect_before_me += (1 - int(samples_certainties[i - 1][1]))
        if samples_certainties[i][1]:
            # if i'm correct at this sample, i agree with all the incorrect that are to come
            n_s += incorrect_after_me[i]
            # i disagree with all the incorrect that preceed me
            n_d += incorrect_before_me
        else:
            # else i'm incorrect, so i disagree with all the correct that are to come
            n_d += (total_samples - i - 1) - incorrect_after_me[i]  # (total_samples - i - 1) = all samples after me
            # and agree with all the correct that preceed me
            n_s += i - incorrect_before_me

    # g_correlation = (n_s - n_d) / (n_s + n_d)
    results = {}
    results['gamma'] = (n_s - n_d) / (n_s + n_d)
    results['AUROC'] = (n_s) / (n_s + n_d)
    results['n_s'] = n_s
    results['n_d'] = n_d
    return results


def calc_OOD_metrics(severity_levels_classes, confidences, confidence_type):
    """
    given attack_groups where each row is a group
    :param severity_levels_classes:
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

    sample_certainties = torch.from_numpy(np.stack([id_confs, id_is_correct], axis=1))

    id_gamma_results = gamma_correlation(sample_certainties, sort=True)

    severity_levels_results = []

    for severity_level_classes in severity_levels_classes:
        # severity_level_classes is a vector of class ids that belong to the groups
        attack_samples = np.any(severity_level_classes[:, np.newaxis] == labels[np.newaxis, :], axis=0)

        attack_conf = confs[attack_samples]
        attack_avg_conf = np.mean(attack_conf)

        attack_is_correct = is_correct[attack_samples]
        assert np.sum(attack_is_correct) == 0

        certainties = np.concatenate([attack_conf, id_confs])
        is_id = np.concatenate([np.zeros_like(attack_conf), np.ones_like(id_confs)])

        assert len(is_id) == len(certainties)

        sample_certainties = torch.from_numpy(np.stack([certainties, is_id], axis=1))
        idx = torch.randperm(sample_certainties.shape[0])
        sample_certainties = sample_certainties[idx]
        gamma_results = gamma_correlation(sample_certainties, sort=True)

        results = {'ood-auroc': gamma_results['AUROC'], 'ood_avg_conf': attack_avg_conf,
                   'id-acc': id_acc, 'id-auroc': id_gamma_results['AUROC'], 'id_avg_conf': id_avg_conf,

                   }

        severity_levels_results.append(results)

    return severity_levels_results
