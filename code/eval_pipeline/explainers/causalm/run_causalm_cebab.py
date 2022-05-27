#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
Estimating TreATEs on OpenTable, using CausaLM """

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import numpy as np
import transformers
from datasets import load_metric
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    default_data_collator,
    set_seed,
    Trainer,
    AutoModel,
)
from transformers.trainer_utils import get_last_checkpoint

from data import DataCollatorForCausalmAdditionalPretraining
from modeling import (
    CausalmHeadConfig,
    LSTMConfig,
    LSTMModel,

    LSTMCausalmConfig,
    BertCausalmConfig,
    RobertaCausalmConfig,
    GPT2CausalmConfig,

    LSTMForCausalmAdditionalPreTraining,
    BertForCausalmAdditionalPreTraining,
    RobertaForCausalmAdditionalPreTraining,
    GPT2ForCausalmAdditionalPreTraining,

    LSTMCausalmForNonlinearSequenceClassification,
    BertCausalmForNonlinearSequenceClassification,
    RobertaCausalmForSequenceClassification,
    GPT2CausalmForNonlinearSequenceClassification, LSTMCausalmForSequenceClassification,
)
from trainer import CausalmTrainer, CausalmTrainingArguments
from utils import (
    get_cebab,
    preprocess_cebab_for_causalm,
    LABEL_COL,
    WANDB_PROJECT,
    CAUSALM_TASK_LABEL_COL,
    task_to_keys,
    TRANSFORMERS_CACHE,
    id2label_dicts,
    BERT,
    ROBERTA,
    ROBERTA_VOCAB_SIZE,
    LSTM,
    GPT2, PROJECT_DIR
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    group: Optional[str] = field(
        default=None,
        metadata={'help': 'The group name to appear in WANDB.'}
    )
    dataset_path: str = field(
        default='./training_dataset/', metadata={"help": "The path to the root folder of the causal datasets."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    cls: int = field(
        default=1, metadata={"help": "The index of the class to measure the ATE with respect to (defaults to 1)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_path is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_architecture: Optional[str] = field(
        metadata={"help": "The base model architecture, for example: 'bert-base-uncased'"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    fasttext_embeddings_path: Optional[str] = field(
        default=str(PROJECT_DIR / 'utils' / 'lstm_embeddings.bin'), metadata={"help": "The path to the LSTM embeddings"}
    )
    tc_heads_types: List[str] = field(
        default=None,
        metadata={"help": f"A list of head types for the treatment concept."}
    )
    cc_heads_types: List[str] = field(
        default=None,
        metadata={"help": f"A list of head types for the control concept."}
    )
    tc_heads_num_labels: List[int] = field(
        default=None,
        metadata={"help": "The number of labels the TC head predicts. Defaults to 2."}
    )
    cc_heads_num_labels: List[int] = field(
        default=None,
        metadata={"help": "The number of labels the CC head predicts. Defaults to 2."}
    )
    tc_lambda: float = field(
        default=0.2,
        metadata={"help": "The relative weight of the treated concept head. Defaults to 0.2."}
    )
    cc_name: str = field(
        default=None,
        metadata={"help": "The name of the CC to infer label2id."}
    )

    def __post_init__(self):
        if self.tc_heads_types is None:
            self.tc_heads_types = list()
        if self.cc_heads_types is None:
            self.cc_heads_types = list()

        if self.tc_heads_num_labels is None:
            self.tc_heads_num_labels = list()
        if self.cc_heads_num_labels is None:
            self.cc_heads_num_labels = list()

        if len(self.tc_heads_types) != len(self.tc_heads_num_labels):
            raise RuntimeError('Treatment heads input error. len(tc_heads_types) must match len(tc_heads_num_labels)')
        if len(self.cc_heads_types) != len(self.cc_heads_num_labels):
            raise RuntimeError('Control heads input error. len(cc_heads_types) must match len(cc_heads_num_labels)')
        if len(self.tc_heads_types) + len(self.cc_heads_types) == 0:
            warnings.warn('No CausaLM heads were detected!')


def main():
    os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CausalmTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, causalm_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, causalm_training_args = parser.parse_args_into_dataclasses()

    if data_args.group:
        os.environ["WANDB_RUN_GROUP"] = data_args.group

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = causalm_training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    sub_output_dir = None
    if causalm_training_args.counterfactual_training:
        if causalm_training_args.causalm_additional_pretraining:
            sub_output_dir = 'additional_pretraining'
        if causalm_training_args.do_train:
            sub_output_dir = 'downstream'
        causalm_training_args.output_dir = os.path.join(causalm_training_args.output_dir, sub_output_dir)
    causalm_training_args.run_name = causalm_training_args.run_name = f'{sub_output_dir}__seed_{causalm_training_args.seed}'

    logger.info(f"WANDB RUN NAME: {causalm_training_args.run_name}")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {causalm_training_args.local_rank}, device: {causalm_training_args.device}, n_gpu: {causalm_training_args.n_gpu}"
        + f"distributed training: {bool(causalm_training_args.local_rank != -1)}, 16-bits training: {causalm_training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {causalm_training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(causalm_training_args.output_dir) \
            and (causalm_training_args.do_train or causalm_training_args.causalm_additional_pretraining) \
            and not causalm_training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(causalm_training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(causalm_training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({causalm_training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and causalm_training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(causalm_training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    is_regression = False
    if data_args.task_name is not None:
        raw_datasets = get_cebab(data_args.task_name)
        raw_datasets = raw_datasets.rename_column(LABEL_COL, CAUSALM_TASK_LABEL_COL)
        if causalm_training_args.causalm_additional_pretraining:
            raw_datasets = preprocess_cebab_for_causalm(
                raw_datasets,
                causalm_training_args.tc_labels_col,
                causalm_training_args.cc_labels_col,
            )
        label_list = sorted(set(raw_datasets['train'][CAUSALM_TASK_LABEL_COL]))
        num_labels = len(label_list)
    else:
        raise NotImplementedError('You must provide a task name.')

    config_kwargs = dict()
    # determine model architecture and instantiate config and model accordingly
    if model_args.model_architecture == BERT:
        config_class = BertCausalmConfig
        additional_pretraining_model_class = BertForCausalmAdditionalPreTraining
        task_model_class = BertCausalmForNonlinearSequenceClassification
        mlm = True  # for data collator - bert is a masked language model

    elif model_args.model_architecture == GPT2:
        config_class = GPT2CausalmConfig
        additional_pretraining_model_class = GPT2ForCausalmAdditionalPreTraining
        task_model_class = GPT2CausalmForNonlinearSequenceClassification
        mlm = False  # for data collator - gpt2 is a causal language model

    elif model_args.model_architecture == ROBERTA:
        config_class = RobertaCausalmConfig
        additional_pretraining_model_class = RobertaForCausalmAdditionalPreTraining
        task_model_class = RobertaCausalmForSequenceClassification  # RobertaClassificationHead is non-linear
        mlm = True  # for data collator - roberta can be masked language model

    elif model_args.model_architecture == LSTM:
        model_args.tokenizer_name = BERT  # when using our customized LSTM we need to use bert tokenizer
        config_kwargs['fasttext_embeddings_path'] = model_args.fasttext_embeddings_path
        config_class = LSTMCausalmConfig
        additional_pretraining_model_class = LSTMForCausalmAdditionalPreTraining
        task_model_class = LSTMCausalmForNonlinearSequenceClassification
        mlm = True  # for data collator doesn't affect behavior in the case of LSTM

    else:
        raise RuntimeError(f'Unsupported architecture "{model_args.model_architecture}"')

    additional_pretraining_model = None
    task_model = None

    # CausaLM additional pretraining
    if causalm_training_args.counterfactual_training and causalm_training_args.causalm_additional_pretraining:
        additional_pretraining_config = config_class(
            tc_heads_cfg=[CausalmHeadConfig(head_type=head_type, head_name=f'tc_head_{idx}', head_params={'num_labels': num_labels})
                          for idx, (head_type, num_labels) in enumerate(zip(model_args.tc_heads_types, model_args.tc_heads_num_labels))
                          ],
            cc_heads_cfg=[CausalmHeadConfig(head_type=head_type, head_name=f'cc_head_{idx}', head_params={'num_labels': num_labels},
                                            id2label=id2label_dicts[model_args.cc_name] if model_args.cc_name else None)
                          for idx, (head_type, num_labels) in enumerate(zip(model_args.cc_heads_types, model_args.cc_heads_num_labels))
                          ],
            tc_lambda=model_args.tc_lambda,
            **config_kwargs
        )
        if model_args.model_architecture == ROBERTA:
            additional_pretraining_config.vocab_size = ROBERTA_VOCAB_SIZE
        additional_pretraining_model = additional_pretraining_model_class(additional_pretraining_config)

    # CausaLM finetuning on downstream task
    elif causalm_training_args.counterfactual_training and not causalm_training_args.causalm_additional_pretraining:
        additional_pretraining_config = config_class.from_pretrained(model_args.model_name_or_path)
        additional_pretraining_model = additional_pretraining_model_class(additional_pretraining_config)
        task_config = config_class(
            model_type=model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            sequence_classifier_type='task',
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            **config_kwargs
        )
        task_model = task_model_class(task_config, additional_pretraining_model.get_counterfactual_weights())

    # Regular training, not related to CausaLM by any means
    else:
        if model_args.model_architecture != LSTM:
            model = AutoModel.from_pretrained(model_args.model_name_or_path)
        else:
            model = LSTMModel(
                config=LSTMConfig(model_args.fasttext_embeddings_path)
            )
        task_config = config_class(
            model_type=model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            sequence_classifier_type='task',
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            **config_kwargs
        )
        task_model = task_model_class(task_config, model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_architecture == GPT2:
        # Define a padding token
        tokenizer.pad_token = tokenizer.eos_token
        if additional_pretraining_model:
            additional_pretraining_model.config.pad_token_id = tokenizer.pad_token_id
        if task_model:
            task_model.config.pad_token_id = tokenizer.pad_token_id

    # Preprocessing the raw_datasets
    if data_args.task_name in task_to_keys:
        text_key, _ = task_to_keys[data_args.task_name]
    else:
        raise RuntimeError(f'Unsupported task {data_args.task_name}.')

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            not causalm_training_args.causalm_additional_pretraining
            and task_model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in task_model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        task_model.config.label2id = label_to_id
        task_model.config.id2label = {id: label for label, id in task_config.label2id.items()}
    elif not causalm_training_args.causalm_additional_pretraining and data_args.task_name is not None and not is_regression:
        task_model.config.label2id = {l: i for i, l in enumerate(label_list)}
        task_model.config.id2label = {id: label for label, id in task_config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        result = tokenizer(examples[text_key], padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[lbl] if lbl != -1 else -1) for lbl in examples["label"]]
        return result

    with causalm_training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if causalm_training_args.do_train or causalm_training_args.causalm_additional_pretraining:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if causalm_training_args.do_eval or causalm_training_args.causalm_additional_pretraining:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if causalm_training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if causalm_training_args.do_train or causalm_training_args.causalm_additional_pretraining:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    accuracy_metric = load_metric('accuracy')

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=1)
        clf_report = classification_report(p.label_ids, preds, digits=5, output_dict=True)
        result = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        result["Macro-F1"] = clf_report["macro avg"]["f1-score"]
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        task_data_collator = default_data_collator
    elif causalm_training_args.fp16:
        task_data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        task_data_collator = None

    # create CausaLM Data Collator
    additional_pretraining_data_collator = DataCollatorForCausalmAdditionalPretraining(
        tokenizer=tokenizer,
        mlm_probability=causalm_training_args.mlm_probability,
        mlm=mlm
    )

    # Additional pretraining
    if causalm_training_args.causalm_additional_pretraining:
        # create CausaLM Trainer
        additional_pretraining_trainer = CausalmTrainer(
            additional_pretraining_model,
            causalm_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=additional_pretraining_data_collator,
            tokenizer=tokenizer,
        )

        # Additional pretraining (using causalm adversarial heads)
        additional_pretraining_trainer.train()
        additional_pretraining_trainer.save_model()
        task_trainer = None
    else:
        task_trainer = Trainer(
            model=task_model,
            args=causalm_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=task_data_collator,
        )

    # Task Training
    if causalm_training_args.do_train:
        checkpoint = None
        if causalm_training_args.resume_from_checkpoint is not None:
            checkpoint = causalm_training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = task_trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        task_trainer.save_model()  # Saves the tokenizer too for easy upload

        task_trainer.log_metrics("train", metrics)
        task_trainer.save_metrics("train", metrics)
        task_trainer.save_state()

    # Evaluation
    if causalm_training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_datasets = {'eval': eval_dataset}
        for dataset_name, dataset in eval_datasets.items():
            metric_key_prefix = dataset_name

            metrics = task_trainer.evaluate(eval_dataset=dataset, metric_key_prefix=metric_key_prefix)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(dataset))

            task_trainer.log_metrics(dataset_name, metrics)
            task_trainer.save_metrics(dataset_name, metrics)

    # Task Prediction
    if causalm_training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = task_trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(causalm_training_args.output_dir, f"predict_results_{task}.txt")
            if task_trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    # if data_args.task_name is not None:
    #     kwargs["language"] = "en"
    #     kwargs["dataset_tags"] = "glue"
    #     kwargs["dataset_args"] = data_args.task_name
    #     kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"
    #
    # if task_trainer:
    #     if causalm_training_args.push_to_hub:
    #         task_trainer.push_to_hub(**kwargs)
    #     else:
    #         task_trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
