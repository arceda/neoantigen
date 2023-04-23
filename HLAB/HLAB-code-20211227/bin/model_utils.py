import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.nn import MSELoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING
import math

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert+RNN
    """,
    BERT_START_DOCSTRING,
)
class ProteinBertSequenceClsRnn(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden
        self.max_seq_len = config.length


        self.bert = BertModel(config)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                              num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError

        self.dropout = nn.Dropout(self.rnn_dropout)
        self.classifier = nn.Linear(2*self.rnn_hidden, self.config.num_labels)

        self.init_weights()

        reduction = 'mean'

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        token_type_ids = None

        bert_out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # lstm
        if self.rnn_type == "lstm":
            pooled_output = bert_out[0]
            rnn_out, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            pooled_output = bert_out[0]
            rnn_out, ht = self.rnn(pooled_output)
        else:
            raise ValueError

        output = rnn_out.permute(0, 2, 1)
        output = torch.nn.functional.max_pool1d(output, self.max_seq_len)
        model_output = self.dropout(output.squeeze())
        logits = self.classifier(model_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = self.criterion
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + (model_output,)
            res=(loss,) + output
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions,
        )




@add_start_docstrings(
    """
    Bert+RNN+Attention
    """,
    BERT_START_DOCSTRING,
)
class ProteinBertSequenceClsRnnAtt(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden
        self.max_seq_len = config.length
        self.att_dropout = 0.1

        self.bert = BertModel(config)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                              num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError

        self.w_omega = nn.Parameter(torch.Tensor(
            self.rnn_hidden * 2, self.rnn_hidden * 2))
        self.u_omega = nn.Parameter(torch.Tensor(self.rnn_hidden * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.dropout = nn.Dropout(self.rnn_dropout)
        self.att_dropout = nn.Dropout(self.att_dropout)
        self.classifier = nn.Linear(2*self.rnn_hidden, self.config.num_labels)

        self.init_weights()

        reduction = 'mean'

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) /math.sqrt(d_k)  #   scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)
        rattention = torch.matmul(p_attn, x)
        context = torch.matmul(p_attn, x).sum(1)  # [batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, rattention

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        token_type_ids = None

        bert_out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # lstm
        if self.rnn_type == "lstm":
            pooled_output = bert_out[0]
            rnn_out, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            pooled_output = bert_out[0]
            rnn_out, ht = self.rnn(pooled_output)
        else:
            raise ValueError

        query = self.att_dropout(rnn_out)
        attn_output, attention = self.attention_net(rnn_out, query)  # 和LSTM的不同就在于这一句
        logits = self.classifier(attn_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = self.criterion
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + attention[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions,
        )




class ProteinBertSequenceClsCnn(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cnn_filters = config.cnn_filters
        self.cnn_dropout = config.cnn_dropout
        self.kernel_size = [1,3,5]

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.cnn_dropout)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.cnn_filters, (K, self.hidden_size)) for K in self.kernel_size])
        self.classifier = nn.Linear(len(self.kernel_size) * self.cnn_filters, self.num_labels)
        self.maxpool = nn.MaxPool1d(3, stride=3)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]

        bert_out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #bert_out[0]   (batch_size,sequence_lenth,hidden_size)
        outputs = bert_out[0].unsqueeze(1)     # (batch_size,1,sequence_lenth,hidden_size)
        outputs = [F.relu(conv(outputs)).squeeze(3) for conv in self.convs]  # [(batch_size, cnn_filters, sequence_lenth-K+1), ...]*len(Ks)
        outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in outputs]  # [(batch_size, cnn_filters), ...]*len(Ks)
        outputs = torch.cat(outputs, 1)
        outputs = self.dropout(outputs)  # (batch_size, len(Ks)*cnn_filters)
        logits = self.classifier(outputs)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions,
        )
