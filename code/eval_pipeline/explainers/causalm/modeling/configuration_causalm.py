from transformers import PretrainedConfig

SEQUENCE_CLASSIFICATION = 'seq_classification'
TOKEN_CLASSIFICATION = 'token_classification'

HEAD_TYPES = {SEQUENCE_CLASSIFICATION, TOKEN_CLASSIFICATION}

__all__ = ['CausalmHeadConfig']


class CausalmHeadConfig(PretrainedConfig):
    model_type: str = "causalm_head"
    is_composition: bool = False

    def __init__(self, head_type=None, head_name=None, head_params: dict = None, **kwargs):
        super(CausalmHeadConfig, self).__init__(**kwargs)

        if head_type is not None and head_type not in HEAD_TYPES:
            raise ValueError(f'Illegal head type: "{head_type}"')

        if not head_params:
            head_params = dict()

        self.head_name = head_name
        self.head_type = head_type

        if head_type == SEQUENCE_CLASSIFICATION or head_type == TOKEN_CLASSIFICATION:
            self.hidden_dropout_prob = head_params.pop('hidden_dropout_prob', 0.0)
            self.num_labels = head_params.pop('num_labels', 2)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head_name}) {self.to_json_string(use_diff=True)}"
