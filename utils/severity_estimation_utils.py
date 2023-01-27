import numpy as np


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


def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), 'valid') / window_size


def diversity_severity_alg(severity_proxy, num_attack_groups):
    # this is an easter egg algorithm (not mentioned in the paper)
    # it gives you more diverse groups of classes to consider as severity levels.

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


def percentile_severity_alg(groups_severity, num_groups):
    percentiles = np.linspace(0, 1, num_groups)
    chosen_attack_groups_ids = [int(q * len(groups_severity)) for q in percentiles]
    chosen_attack_groups_ids[-1] -= 1
    attack_severities = groups_severity[chosen_attack_groups_ids]

    return chosen_attack_groups_ids, attack_severities, percentiles


def get_severity_levels_groups_of_classes(severity_proxy, num_severity_levels, num_classes_per_group,
                                          choice_alg='fixed_percentiles'):
    cood_classes_ids = np.argsort(severity_proxy)
    severity_proxy = np.sort(severity_proxy)

    groups_severity = moving_average(severity_proxy, num_classes_per_group)

    if choice_alg == 'diversity':
        chosen_groups_starts, severities_scores, percentiles = diversity_severity_alg(groups_severity,
                                                                                          num_severity_levels)
    elif choice_alg == 'fixed_percentiles':
        chosen_groups_starts, severities_scores, percentiles = percentile_severity_alg(groups_severity,
                                                                                           num_severity_levels)
    else:
        raise ValueError('given choice alg is not supported')

    severity_levels_groups = np.array([cood_classes_ids[idx: idx + num_classes_per_group] for idx in chosen_groups_starts])

    return {'severity_levels_groups': severity_levels_groups, 'severity_levels_scores': severities_scores,
            'percentiles': percentiles}
