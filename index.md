![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# <img src="https://i.ibb.co/d5bTqrt/Kebab-icon.png" width="40" height="40"> CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior
<p align="center">

</p>

## What is <img src="https://i.ibb.co/d5bTqrt/Kebab-icon.png" width="30" height="30"> CEBaB?
✅ English-language benchmark to evaluate causal explanation methods.  
✅ Human-validated Aspect-based Sentiment Analysis (ABSA) benchmark.   


## Contents

* [Citation](#citation)
* [Dataset files](#dataset-files)
* [Datasheet](#datasheet)
* [Quick start](#quick-start)
* [Data format](#data-format)
* [Code](#code)
* [License](#license)

## Citation

[Eldar David Abraham](https://eldarab.github.io/), [Karel D'Oosterlink](https://www.kareldoosterlinck.com/), [Amir Feder](https://amirfeder.github.io/), [Yair Gat](https://yairgat.github.io/), [Atticus Geiger](https://atticusg.github.io/), [Christopher Potts](http://web.stanford.edu/~cgpotts/), [Roi Reichart](https://iew.technion.ac.il/~roiri/), [Zhengxuan Wu](http://zen-wu.social). 2022. [CEBaB: Estimating the causal effects of real-world concepts on NLP model behavior](https://arxiv.org/abs/2205.14140). Ms., Stanford University, Technion -- Israel Institute of Technology, and Ghent University.

{% raw %}
```stex
@unpublished{abraham-etal-2022-cebab,
    title={{CEBaB}: Estimating the Causal Effects of Real-World Concepts on {NLP} Model Behavior},
    author={Abraham, Eldar David and D'Oosterlinck, Karel and Feder, Amir and Gat, Yair Ori and Geiger, Atticus and Potts, Christopher and Reichart, Roi and Wu, Zhengxuan},
    note={arXiv:2205.14140},
    url={https://arxiv.org/abs/2205.14140},
    year={2022}}
```
{% endraw %}

## Dataset files

Dataset files can be downloaded from [CEBaB-v1.1.zip](CEBaB-v1.1.zip). Our v1.1 differs from v1.0 only in that v1.1 has proper unique ids our examples and corrects a bug that led to some non-unique ids in the previous version. There are no changes to other critical fields.

**Note that we recommend you use [HuggingFace Datasets](https://huggingface.co/datasets/CEBaB/CEBaB) library to use our dataset. See below for a 1-linear data loading.**

The dataset consists of train_exclusive/train_inclusive/dev/test splits:

* `train_exclusive.json`
* `train_inclusive.json`
* `train_observational.json`
* `dev.json`
* `test.json`

## Datasheet

The [Datasheet](https://arxiv.org/abs/xxxx.xxxxx) for our dataset:

* [cebab_datasheet.md](cebab_datasheet.md)

## Quick start

### <img src="https://avatars.githubusercontent.com/u/25720743?s=200&v=4" width="20" height="20"> Huggingface (Recommended)

CEBaB is mainly maintained using the [HuggingFace Datasets](https://huggingface.co/datasets/CEBaB/CEBaB) library:
```python
"""
Make sure you install the Datasets library using:
pip install datasets
"""
from datasets import load_dataset

CEBaB = load_dataset("CEBaB/CEBaB")
```

### Local Files (Not Recommended)

This function can be used to load any subset of the raw `*.json` files:

```python
import json

def load_split(splitname):
    with open(splitname) as f:
        data = json.load(f)
    return data
```

## Data format

```python
{    
    'id': str in format dddddd_dddddd as the concatenation of original_id and edit_id,
    'original_id': str in format dddddd,
    'edit_id': str in format dddddd,
    'is_original': bool,
    'edit_goal': str (one of "Negative", "Positive", "unknown") or None if is_original,
    'edit_type': str (one of "noise", "service", "ambiance", "food"),
    'edit_worker': str or None if is_original,
    'description': str,
    'review_majority': str (one of "1", "2", "3", "4", "5", "no majority"),
    'review_label_distribution': dict (str to int),
    'review_workers': dict (str to str),
    'food_aspect_majority': str (one of "Negative", "Positive", "unknown", "no majority"),
    'ambiance_aspect_majority': str (one of "Negative", "Positive", "unknown", "no majority"),
    'service_aspect_majority': str (one of "Negative", "Positive", "unknown", "no majority"),
    'noise_aspect_majority': str (one of "Negative", "Positive", "unknown", "no majority"),
    'food_aspect_label_distribution': dict (str to int),
    'ambiance_aspect_label_distribution': dict (str to int),
    'service_aspect_label_distribution': dict (str to int),
    'noise_aspect_label_distribution': dict (str to int),
    'food_aspect_validation_workers': dict (str to str),
    'ambiance_aspect_validation_workers': dict (str to str),
    'service_aspect_validation_workers': dict (str to str),
    'noise_aspect_validation_workers': dict (str to str),
    'opentable_metadata': {
        "restaurant_id": int,
        "restaurant_name": str,
        "cuisine": str,
        "price_tier": str,
        "dining_style": str,
        "dress_code": str,
        "parking": str,
        "region": str,
        "rating_ambiance": int,
        "rating_food": int,
        "rating_noise": int,
        "rating_service": int,
        "rating_overall": int
    }
}
```

Details:

* `'id'`: The unique identifier this example (an combination of two ids listed below).
* `'original_id'`: The unique identifier of the original sentence for an edited example.
* `'edit_id'`: The unique identifier of the edited sentence.
* `'is_original'`: Indicate whether this sentence is an edit or not.
* `'edit_goal'`: The goal label for the editing aspect if it an edited example, else `None`.
* `'edit_type'`: The aspect to modify or to label with sentiment if it an edited example, else `None`.
* `'edit_worker'`: Anonymized MTurk id of the worker who wrote `'description'`. These are from the same family of ids as used in `'aspect_validation_workers'`.
* `'description'`: The example text.
* `'review_majority'`: The review-level label for the editing aspect chosen by at least three of the five workers if there is one, else `no majority`.
* `'review_label_distribution'`: Review-level rating distribution from the MTurk validation task.
* `'review_workers'`: Individual response for review-level rating from annotators. The keys are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'*_aspect_majority'`: The aspect-level label for the editing aspect chosen by at least three of the five workers if there is one, else `no majority`.
* `'*_aspect_label_distribution'`: Aspect-level rating distribution from the MTurk validation task.
* `'*_aspect_label_workers'`: Individual response for review-level rating from annotators. The keys are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'opentable_metadata'`: Metadata for the review.

Here is one example,

```python
{
    "id": "000000_000000",
    "original_id": "000000",
    "edit_id": "000000",
    "is_original": true,
    "edit_goal": null,
    "edit_type": null,
    "edit_worker": null,
    "description": "Overbooked and didnot honor reservation time,put on wait list with walk INS",
    "review_majority": "1",
    "review_label_distribution": {
        "1": 4,
        "2": 1
    },
    "review_workers": {
        "w244": "1",
        "w120": "2",
        "w197": "1",
        "w7": "1",
        "w132": "1"
    },
    "food_aspect_majority": "",
    "ambiance_aspect_majority": "",
    "service_aspect_majority": "Negative",
    "noise_aspect_majority": "unknown",
    "food_aspect_label_distribution": "",
    "ambiance_aspect_label_distribution": "",
    "service_aspect_label_distribution": {
        "Negative": 5
    },
    "noise_aspect_label_distribution": {
        "unknown": 4,
        "Negative": 1
    },
    "food_aspect_validation_workers": "",
    "ambiance_aspect_validation_workers": "",
    "service_aspect_validation_workers": {
        "w148": "Negative",
        "w120": "Negative",
        "w83": "Negative",
        "w35": "Negative",
        "w70": "Negative"
    },
    "noise_aspect_validation_workers": {
        "w27": "unknown",
        "w23": "unknown",
        "w81": "Negative",
        "w103": "unknown",
        "w9": "unknown"
    },
    "opentable_metadata": {
        "restaurant_id": 6513,
        "restaurant_name": "Molino's Ristorante",
        "cuisine": "italian",
        "price_tier": "low",
        "dining_style": "Casual Elegant",
        "dress_code": "Smart Casual",
        "parking": "Private Lot",
        "region": "south",
        "rating_ambiance": 1,
        "rating_food": 3,
        "rating_noise": 2,
        "rating_service": 2,
        "rating_overall": 2
    }
}
```

## Code

We host our analyses code at our [code folder](https://github.com/CEBaBing/CEBaB).

## Leaderboard

This section contains the leaderboard for some of the best scores obtained on CEBaB as a five-class sentiment classification task. To add scores please consider a pull request.

Model Architecture | Metric | Approx | S-Learner | INLP
--- | --- | --- | --- | --- 
BERT | L2 | 0.81 (± 0.01) | 0.74 (± 0.02) | 0.80 (± 0.02)
BERT | COS | 0.61 (± 0.01) | 0.63 (± 0.01) | 0.59 (± 0.03)
BERT | NormDiff | 0.44 (± 0.01) | 0.54 (± 0.02) | 0.73 (± 0.02)
RoBERTa | L2 | 0.83 (± 0.01) | 0.78 (± 0.01) | 0.84 (± 0.01)
RoBERTa | COS | 0.60 (± 0.01) | 0.64 (± 0.01) | 0.58 (± 0.01)
RoBERTa | NormDiff | 0.45 (± 0.00) | 0.59 (± 0.01) | 0.81 (± 0.01)
GPT-2 | L2 | 0.72 (± 0.02) | 0.60 (± 0.02) | 0.72 (± 0.01)
GPT-2 | COS | 0.59 (± 0.01) | 0.59 (± 0.01) | 1.00 (± 0.00)
GPT-2 | NormDiff | 0.41 (± 0.01) | 0.40 (± 0.01) | 0.58 (± 0.03)
LSTM | L2 | 0.86 (± 0.01) | 0.73 (± 0.01) | 0.79 (± 0.01)
LSTM | COS | 0.64 (± 0.01) | 0.64 (± 0.01) | 0.74 (± 0.02)
LSTM | NormDiff | 0.50 (± 0.01) | 0.53 (± 0.01) | 0.60 (± 0.01)

## License

CeBaB has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
