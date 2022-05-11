![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# <img src="https://i.ibb.co/d5bTqrt/Kebab-icon.png" width="60" height="60"> CEBaB: A Causal Benchmark Dataset for Concept-based Explanation Methods in NLP
<p align="center">

</p>

<div align="center">
  <img src="https://i.ibb.co/W2pr2w6/explanation-scheme.png" style="float:left" width="800px">
</div>
<p></p>

## What is <img src="https://i.ibb.co/d5bTqrt/Kebab-icon.png" width="40" height="40"> CEBaB?
:white_check_mark: It is an English-language benchmark task that supports a wide range of **causal analyses** due to its very rich counterfactual structure.   
:white_check_mark: It is a human-validated **Aspect-based Sentiment Analysis (ABSA) benchmark**.   


## Contents

* [Citation](#citation)
* [Dataset files](#dataset-files)
* [Quick start](#quick-start)
* [Data format](#data-format)
* [Benchmarking for Causal Effects](#benchmarking-for-causal-effects)
* [License](#license)

## Citation

## Dataset files

The dataset is [CEBaB-v1.0.zip](CEBaB-v1.0.zip), which is included in this repository.

## Quick start

### <img src="https://avatars.githubusercontent.com/u/25720743?s=200&v=4" width="30" height="30"> Huggingface (Recommended)

CEBaB is mainly maintained using the [HuggingFace Datasets](https://huggingface.co/datasets/CEBaB/CEBaB) library:
```python
"""
Make sure you install the Datasets library using:
pip install datasets
"""
from datasets import load_dataset

CEBaB = load_dataset("CEBaB/CEBaB")
```

### Local Files

This function can be used to load any subset of the files:

```python
import json

def load_dataset(*src_filenames):
    data = []
    for filename in src_filenames:
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                data.append(d)
    return data
```

## Data format

```python
{    
    "id": str in format dddddd_dddddd as the concatenation of original_id and edit_id,
    "original_id": str in format dddddd,
    "edit_id: str in format dddddd,
    "is_original" bool,
    "aspect": str (one of "noise evaluation", "service evaluation", "ambiance evalution", "food evaluation")
    "edit_goal": str (one of "Negative", "Positive", "unknown") or None if is_original,
    "edit_worker": str or None if is_original,
    "description": str,
    "aspect_majority": str (one of "Negative", "Positive", "unknown", "no majority"),
    "aspect_label_distribution": dict (str to int),
    "aspect_validation_workers": dict (str to int),
    "review_majority": str (one of "1", "2", "3", "4", "5", "no majority"),
    "review_label_distribution": dict (str to int),
    "review_validation_workers": dict (str to int),
    "opentable_metadata": {
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
* `'aspect'`: The aspect to modify or to label with sentiment.
* `'edit_goal'`: The goal label for the editing aspect if it an edited example, else `None`.
* `'edit_worker'`: Anonymized MTurk id of the worker who wrote `'description'`. These are from the same family of ids as used in `'aspect_validation_workers'`.
* `'description'`: The example text.
* `'aspect_majority'`: The aspect-level label for the editing aspect chosen by at least three of the five workers if there is one, else `no majority`.
* `'aspect_label_distribution'`: Aspect-level rating distribution from the MTurk validation task.
* `'aspect_validation_workers'`: Individual response for aspect-level rating from annotators. The keys are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'review_majority'`: The review-level label for the editing aspect chosen by at least three of the five workers if there is one, else `no majority`.
* `'review_label_distribution'`: Review-level rating distribution from the MTurk validation task.
* `'review_validation_workers'`: Individual response for review-level rating from annotators. The keys are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'opentable_metadata'`: Metadata for the review.

Here is one example,

```python
{
    "id": "000001_000001",
    "original_id": "000001",
    "edit_id": "000001",
    "is_original": false,
    "aspect": "noise",
    "edit_goal": "Negative",
    "edit_worker": "w82",
    "description": "Overbooked and didnot honor reservation time,put on wait list with walk INS. Overly loud in dining area.",
    "aspect_majority": "Negative",
    "aspect_label_distribution": {
        "Negative": 5
    },
    "aspect_validation_workers": {
        "w16": "Negative",
        "w63": "Negative",
        "w36": "Negative",
        "w7": "Negative",
        "w172": "Negative"
    },
    "review_majority": "no majority",
    "review_label_distribution": {
        "1": 2,
        "2": 2,
        "3": 1
    },
    "review_validation_workers": {
        "w152": "2",
        "w16": "1",
        "w44": "1",
        "w64": "2",
        "w32": "3"
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

## Benchmarking for Causal Effects
Equipped with CEBaB, we conduct experiments to assess the quality of explanations generated by concept-based explanation methods, to explain the predictions of a language model.

| Treatment | Ground-truth| CausalLM | INLP | TCAV | ConceptSHAP | 
| :---:     | :---:       |    :---: | :---:| :---:| :---:       |

## License

CeBaB has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
