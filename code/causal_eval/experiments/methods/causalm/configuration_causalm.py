from typing import List, Dict, Any

from transformers import PretrainedConfig
from transformers.models.bert.configuration_bert import BertConfig

from .utils import HEAD_TYPES, SEQUENCE_CLASSIFICATION, TOKEN_CLASSIFICATION

__all__ = ['CausalmHeadConfig', 'BertCausalmConfig']


class CausalmHeadConfig(PretrainedConfig):
    model_type: str = "causalm_head"
    is_composition: bool = False

    def __init__(self, head_name=model_type, head_type=SEQUENCE_CLASSIFICATION, head_params: dict = None, **kwargs):
        super(CausalmHeadConfig, self).__init__(**kwargs)

        if head_type not in HEAD_TYPES:
            raise ValueError(f'Illegal head type: "{head_type}"')

        if not head_params:
            head_params = dict()

        self.head_name = head_name
        self.head_type = head_type

        if head_type == SEQUENCE_CLASSIFICATION or head_type == TOKEN_CLASSIFICATION:
            self.hidden_dropout_prob = head_params.pop('hidden_dropout_prob', 0.0)
            self.num_labels = head_params.pop('num_labels', 2)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head_name}) {self.to_json_string(use_diff=True)}"


class BertCausalmConfig(BertConfig):
    """
    Adds a tc_heads and cc_heads parameters to config.
    """
    model_type = "bert_causalm"

    def __init__(
            self,
            tc_heads_cfg: List[CausalmHeadConfig] = None,
            cc_heads_cfg: List[CausalmHeadConfig] = None,
            tc_lambda: float = 0.2,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.tc_heads_cfg = tc_heads_cfg if tc_heads_cfg else []
        self.cc_heads_cfg = cc_heads_cfg if cc_heads_cfg else []
        self.tc_lambda = tc_lambda
        self.sequence_classifier_type = kwargs.pop("sequence_classifier_type", None)
        self.token_classifier_type = kwargs.pop("token_classifier_type", None)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(use_diff=True)}"

    def to_diff_dict(self) -> Dict[str, Any]:
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BertConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                    key not in default_config_dict
                    or key == "transformers_version"
                    or value != default_config_dict[key]
                    or (key in class_config_dict and value != class_config_dict[key])
            ):
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], CausalmHeadConfig):
                        serializable_config_dict[key] = [head.to_diff_dict() for head in value]
                        # for head_cfg in serializable_config_dict[key]:
                        #     head_cfg.pop('transformers_version')
                else:
                    serializable_config_dict[key] = value

        return serializable_config_dict
