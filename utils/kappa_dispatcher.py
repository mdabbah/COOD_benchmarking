from utils.confidence_functions import extract_softmax_on_dataset, extract_entropy_on_dataset, \
    extract_max_logit_on_dataset, \
    extract_mcd_entropy_on_dataset, extract_odin_confidences_on_dataset


def get_confidence_function(function):
    if callable(function):
        return function

    dispatcher = {'softmax': extract_softmax_on_dataset,
                  'entropy': extract_entropy_on_dataset,
                  'max_logit': extract_max_logit_on_dataset,
                  'MC_dropout_entropy': extract_mcd_entropy_on_dataset,
                  'odin': extract_odin_confidences_on_dataset
                  }

    if function not in dispatcher:
        raise ValueError(f'given confidence function is not callable or not in the set of pre-implemented functions.\n'
                         f'the set of pre-implemented functions are {dispatcher.keys()}')

    return dispatcher[function]

