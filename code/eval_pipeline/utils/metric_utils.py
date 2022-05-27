import numpy as np
from scipy.spatial.distance import cosine


def _calculate_ite(pairs):
    """
    This metric measures the effect of the intervention made by the generating crowd-worker
    on the label the labeling crowd-worker gave. It is independent of the model and the explainer.
    """
    pairs['ITE'] = pairs['review_majority_counterfactual'] - pairs['review_majority_base']

    return pairs


def _calculate_icace(pairs):
    """
    This metric measures the effect of a certain concept on the given model.
    It is independent of the explainer.
    """
    pairs['ICaCE'] = (pairs['prediction_counterfactual'] - pairs['prediction_base']).apply(lambda x: np.round(x, decimals=4))

    return pairs


def _calculate_estimate_loss(pairs):
    """
    Calculate the distance between the ICaCE and EICaCE.
    """

    pairs['ICaCE-L2'] = pairs[['ICaCE', 'EICaCE']].apply(lambda x: np.linalg.norm(x[0] - x[1], ord=2), axis=1)
    pairs['ICaCE-cosine'] = pairs[['ICaCE', 'EICaCE']].apply(lambda x: _cosine_distance(x[0], x[1]), axis=1)
    pairs['ICaCE-normdiff'] = pairs[['ICaCE', 'EICaCE']].apply(lambda x: abs(np.linalg.norm(x[0], ord=2) - np.linalg.norm(x[1], ord=2)), axis=1)

    return pairs


def _cosine_distance(a,b):
    # cosine distance is not defined for zero vectors
    # in this case, return the average distance
    if np.linalg.norm(a, ord=2) == 0 or np.linalg.norm(b, ord=2) == 0:
        return 1
    else:
        return cosine(a,b)


def _aggregate_metrics(pairs, groupby, metrics):
    """
    Aggregates metrics in a dataframe by averaging. Keeps tracks of the counts by summing.
    
    pairs: dataframe to average
    groupby: column names to group by
    metrics: metrics to average 
    """
    # average all metrics, but sum the counts
    metric_to_agg = {metric: ['mean' if metric != 'count' else 'sum'] for metric in metrics}
    # if groupby is empty, aggregate over all data
    if groupby:
        pairs_grouped = pairs.groupby(groupby).agg(metric_to_agg)
        pairs_grouped.columns = metrics
    else:
        pairs_grouped = pairs.agg(metric_to_agg)

    # round metrics
    for metric in metrics:
        if groupby:
            pairs_grouped[f'{metric}'] = pairs_grouped[metric].apply(lambda x: np.round(x, decimals=4))
        else:
            pairs_grouped[f'{metric}'][metric_to_agg[metric][0]] = np.round(pairs_grouped[metric][metric_to_agg[metric][0]], decimals=4)

    return pairs_grouped


def tvd(P: np.ndarray, Q: np.ndarray) -> float:
    r"""
    Credit: https://github.com/rigetti/forest-benchmarking/blob/4c2c3bf94af4926b61e9072ca71b914972de338c/forest/benchmarking/distance_measures.py#L243
    Computes the total variation distance between two (classical) probability
    measures P(x) and Q(x).
    When x is a finITE alphabet then the definition is
    .. math::
        tvd(P,Q) = (1/2) \sum_x |P(x) - Q(x)|
    where tvd(P,Q) is in [0, 1]. There is an alternate definition for non-finITE alphabet measures
    involving a supremum.
    :param P: Is a dim by 1 np.ndarray.
    :param Q: Is a dim by 1 np.ndarray.
    :return: total variation distance which is a scalar.
    """
    rows_p, cols_p = P.shape
    rows_q, cols_q = Q.shape
    if not (cols_p == cols_q == 1 and rows_p > 1 and rows_q > 1):
        raise ValueError("Arrays must be the same length")
    return 0.5 * np.sum(np.abs(P - Q))
