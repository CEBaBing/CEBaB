import pandas as pd

from eval_pipeline.utils import metric_utils, get_intervention_pairs


def cebab_pipeline(model, explainer, train_dataset, dev_dataset, dataset_type='5-way', shorten_model_name=False):
    # get predictions on train and dev
    train_predictions, _ = model.predict_proba(train_dataset)
    dev_predictions, dev_report = model.predict_proba(dev_dataset)

    # append predictions to datasets
    # train_dataset['prediction'] = list(train_predictions)
    dev_dataset['prediction'] = list(dev_predictions)

    # fit explainer
    explainer.fit(train_dataset, train_predictions, model, dev_dataset)

    # get intervention pairs
    pairs_dataset = get_intervention_pairs(dev_dataset, dataset_type=dataset_type)  # TODO why is the index not unique here?

    # mitigate possible data leakage
    allowed_columns = [
        'description_base',
        'review_majority_base',
        'food_aspect_majority_base',
        'service_aspect_majority_base',
        'noise_aspect_majority_base',
        'ambiance_aspect_majority_base',
        'intervention_type',
        'intervention_aspect_base',
        'intervention_aspect_counterfactual',
        'opentable_metadata_base'
    ]

    pairs_dataset_no_leakage = pairs_dataset.copy()[allowed_columns]

    # get explanations
    explanations = explainer.estimate_icace(pairs_dataset_no_leakage)

    # append explanations to the pairs
    pairs_dataset['EICaCE'] = explanations
    # TODO: add cosine
    pairs_dataset = metric_utils._calculate_ite(pairs_dataset)  # effect of crowd-workers on other crowd-workers (no model, no explainer)
    pairs_dataset = metric_utils._calculate_icace(pairs_dataset)  # effect of concept on the model (with model, no explainer)
    pairs_dataset = metric_utils._calculate_estimate_loss(pairs_dataset)  # l2 CEBaB Score (model and explainer)

    # only keep columns relevant for metrics
    CEBaB_metrics_per_pair = pairs_dataset[[
        'intervention_type', 'intervention_aspect_base', 'intervention_aspect_counterfactual', 'ITE', 'ICaCE', 'EICaCE', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff']].copy()
    CEBaB_metrics_per_pair['count'] = 1

    # get CEBaB tables
    metrics = ['count', 'ICaCE', 'EICaCE']

    groupby_aspect_direction = ['intervention_type', 'intervention_aspect_base', 'intervention_aspect_counterfactual']

    CaCE_per_aspect_direction = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, groupby_aspect_direction, metrics)
    CaCE_per_aspect_direction.columns = ['count', 'CaCE', 'ECaCE']
    CaCE_per_aspect_direction = CaCE_per_aspect_direction.set_index(['count'], append=True)
    
    ACaCE_per_aspect = metric_utils._aggregate_metrics(CaCE_per_aspect_direction.abs(), ['intervention_type'], ['CaCE', 'ECaCE'])
    ACaCE_per_aspect.columns = ['ACaCE', 'EACaCE']

    CEBaB_metrics_per_aspect_direction = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, groupby_aspect_direction, ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff'])
    CEBaB_metrics_per_aspect_direction.columns = ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff']
    CEBaB_metrics_per_aspect_direction = CEBaB_metrics_per_aspect_direction.set_index(['count'], append=True)

    CEBaB_metrics_per_aspect = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, ['intervention_type'], ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff'])
    CEBaB_metrics_per_aspect.columns = ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff']
    CEBaB_metrics_per_aspect = CEBaB_metrics_per_aspect.set_index(['count'], append=True)

    CEBaB_metrics = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, [], ['ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff'])

    # get ATE table
    ATE = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, groupby_aspect_direction, ['count', 'ITE'])
    ATE.columns = ['count', 'ATE']

    # add model and explainer information
    if shorten_model_name:
        model_name = str(model).split('.')[0]
    else:
        model_name = str(model)

    CaCE_per_aspect_direction.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) if col != 'CaCE' else (model_name, '', col) for col in CaCE_per_aspect_direction.columns])
    ACaCE_per_aspect.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) if col != 'ACaCE' else (model_name, '', col) for col in ACaCE_per_aspect.columns])
    CEBaB_metrics_per_aspect_direction.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) for col in CEBaB_metrics_per_aspect_direction.columns])
    CEBaB_metrics_per_aspect.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) for col in CEBaB_metrics_per_aspect.columns])
    CEBaB_metrics.index = pd.MultiIndex.from_product([[model_name], [str(explainer)], CEBaB_metrics.index])
    
    # performance report
    performance_report_index = ['macro-f1', 'accuracy']
    performance_report_data = [dev_report['macro avg']['f1-score'], dev_report['accuracy']]
    performance_report_col = [model_name]
    performance_report = pd.DataFrame(data=performance_report_data, index=performance_report_index, columns=performance_report_col)

    return ATE, CEBaB_metrics, CEBaB_metrics_per_aspect_direction, CEBaB_metrics_per_aspect, CaCE_per_aspect_direction, ACaCE_per_aspect, performance_report


def run_pipelines(models, explanators, train, dev, dataset_type='5-way', shorten_model_name=False):
    # run all (model, explainer) pairs
    results_ATE = []
    results_CEBaB_metrics = []
    results_CEBaB_metrics_per_aspect_direction = []
    results_CEBaB_metrics_per_aspect = []
    results_CaCE_per_aspect_direction = []
    results_ACaCE_per_aspect = []
    results_performance_report = []

    for model, explainer in zip(models, explanators):
        print(f'Now running {explainer}')
        train_dataset = train.copy()
        dev_dataset = dev.copy()

        ATE, CEBaB_metrics, CEBaB_metrics_per_aspect_direction, CEBaB_metrics_per_aspect, CaCE_per_aspect_direction, ACaCE_per_aspect, performance_report = cebab_pipeline(
            model, explainer, train_dataset, dev_dataset, dataset_type=dataset_type, shorten_model_name=shorten_model_name)

        results_ATE.append(ATE)
        results_CEBaB_metrics.append(CEBaB_metrics)
        results_CEBaB_metrics_per_aspect_direction.append(CEBaB_metrics_per_aspect_direction)
        results_CEBaB_metrics_per_aspect.append(CEBaB_metrics_per_aspect)
        results_CaCE_per_aspect_direction.append(CaCE_per_aspect_direction)
        results_ACaCE_per_aspect.append(ACaCE_per_aspect)
        results_performance_report.append(performance_report)

    # concat the results
    final_ATE = results_ATE[0]
    final_CEBaB_metrics = pd.concat(results_CEBaB_metrics, axis=0)
    final_CEBaB_per_aspect_direction = pd.concat(results_CEBaB_metrics_per_aspect_direction, axis=1)
    final_CEBaB_per_aspect = pd.concat(results_CEBaB_metrics_per_aspect, axis=1)
    final_CaCE_per_aspect_direction = pd.concat(results_CaCE_per_aspect_direction, axis=1)
    final_ACaCE_per_aspect = pd.concat(results_ACaCE_per_aspect, axis=1)
    final_report = results_performance_report[0]

    # drop duplicate ICaCE columns
    final_CaCE_per_aspect_direction = final_CaCE_per_aspect_direction.loc[:,~final_CaCE_per_aspect_direction.columns.duplicated()]
    final_ACaCE_per_aspect = final_ACaCE_per_aspect.loc[:,~final_ACaCE_per_aspect.columns.duplicated()]


    return final_ATE, final_CEBaB_metrics, final_CEBaB_per_aspect_direction, final_CEBaB_per_aspect, final_CaCE_per_aspect_direction, final_ACaCE_per_aspect, final_report
