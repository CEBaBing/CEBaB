import numpy as np

def dataset_aspects_to_onehot(dataset):
    """
    Encode the aspects of a dataset in a onehot vector.
    """
    # encode the aspect labels
    dataset = dataset.replace('', -1)
    dataset = dataset.replace('no majority', -1)
    dataset = dataset.replace('Negative', 0)
    dataset = dataset.replace('unknown', 1)
    dataset = dataset.replace('Positive', 2)
    reprs = dataset[['food_aspect_majority', 'service_aspect_majority', 'ambiance_aspect_majority', 'noise_aspect_majority']].astype(int).to_numpy()

    # get a mask for the no majority data
    minus_ones = reprs == -1
    reprs_no_minuses = reprs
    reprs_no_minuses[minus_ones] = 0

    # create onehot encodings for the aspect labels
    N = reprs_no_minuses.shape[0]
    reprs_no_minuses = reprs_no_minuses.flatten()
    reprs_onehot = np.zeros((reprs_no_minuses.size, reprs_no_minuses.max() + 1))
    reprs_onehot[np.arange(reprs_no_minuses.size), reprs_no_minuses] = 1
    reprs_onehot = reprs_onehot.reshape(N, -1, 3)

    # use the mask to set no majority data to the zero vector
    reprs_onehot[minus_ones] = np.zeros(reprs_onehot[minus_ones].shape)

    reprs_onehot = reprs_onehot.reshape(reprs_onehot.shape[0], -1)
    return reprs_onehot
