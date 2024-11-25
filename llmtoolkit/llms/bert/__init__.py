from .config_bert import BertConfig
from .modeling_bert import BertModel
from .tasking_bert import BertForMaskedLM, BertForPreTraining

# from .tasks_bert import (BertForMaskedLM, BertForMultipleChoice,
#                          BertForNextSentencePrediction, BertForPreTraining,
#                          BertForQuestionAnswering,
#                          BertForSequenceClassification,
#                          BertForTokenClassification)
# from .tokenization_bert import BertTokenizer

__all__ = ['BertConfig', 'BertModel', 'BertForPreTraining', 'BertForMaskedLM']
