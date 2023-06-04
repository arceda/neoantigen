import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.nn import MSELoss

from tape import ProteinBertAbstractModel, ProteinBertModel

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers import Trainer, TrainingArguments, BertConfig
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING
import math
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TapeLinear(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = ProteinBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):                
        
        outputs = self.bert(input_ids, input_mask=attention_mask) 

        sequence_output, pooled_output = outputs[:2]  

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
      
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))      

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, # TAPE, no esta retornando los hidden states, pero si deberia https://github.com/songlab-cal/tape/blob/master/tape/models/modeling_bert.py
            attentions=None, # TAPE, no esta retornando los attentions pero si deberia
        )

