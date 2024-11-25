import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from llmtoolkit.llms.bert.config_bert import BertConfig
from llmtoolkit.llms.bert.modeling_bert import (BertModel, BertOnlyMLMHead,
                                                BertOnlyNSPHead,
                                                BertPreTrainedModel,
                                                BertPreTrainingHeads)
from llmtoolkit.llms.modeling_outputs import (BertForPreTrainingOutput,
                                              BertModelOutput, CausalLMOutput,
                                              MaskedLMOutput,
                                              MultipleChoiceModelOutput,
                                              NextSentencePredictorOutput,
                                              QuestionAnsweringModelOutput,
                                              SequenceClassifierOutput,
                                              TokenClassifierOutput)


class BertForPreTraining(BertPreTrainedModel):
    """Bert Model with pretraining tasks on top.

    Args:
        config (BertConfig):
            An instance of BertConfig used to construct BertForPreTraining.
    """

    def __init__(self, config: BertConfig):
        super(BertForPreTraining, self).__init__(config)
        self.config = config

        # BERT base model
        self.bert = BertModel(config)

        # Pretraining heads for masked language modeling and next sentence prediction
        self.bert_heads: BertPreTrainingHeads = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.bert_heads.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.bert_heads.lm_head.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        """Forward pass for BertForPretraining.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
            pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor, optional): Attention mask.
            token_type_ids (Tensor, optional): Token type IDs.
            position_ids (Tensor, optional): Positional embeddings.
            head_mask (Tensor, optional): Head mask for attention layers.
            labels (Tensor, optional): Labels for masked language modeling loss.
            next_sentence_label (Tensor, optional): Labels for next sentence prediction.
            output_attentions (bool, optional): Whether to return attentions.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.

        Returns:
            Union[Tuple[torch.Tensor], BertForPreTrainingOutput]: Output tuple or BertForPreTrainingOutput object.
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        # Forward pass through BERT base model
        outputs: BertModelOutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract sequence and pooled outputs from BERT base model
        sequence_output, pooled_output = outputs[:2]

        # Predictions from pretraining heads
        prediction_scores, seq_relationship_score = self.bert_heads(
            sequence_output, pooled_output)

        total_loss = None
        # Calculate total loss if labels are provided for both masked language modeling and next sentence prediction
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            # Return output as a tuple if return_dict is False
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss, ) +
                    output) if total_loss is not None else output

        # Return output as BertForPreTrainingOutput object if return_dict is True
        return BertForPreTrainingOutput(
            loss=total_loss,
            masked_lm_loss=masked_lm_loss,
            next_sentence_loss=next_sentence_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertLMHeadModel(BertPreTrainedModel):
    """Bert Model with a Causal Language Modeling (CLM) head.

    Args:
        config (:class:`BertConfig`):
            An instance of BertConfig used to construct BertLMHeadModel.
    """

    def __init__(self, config):
        """Initializes the BertLMHeadModel.

        Args:
            config (BertConfig): Configuration class for BERT.
        """
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.lm_head = BertOnlyMLMHead(config)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """Retrieves the output embedding layer.

        Returns:
            torch.nn.Module: Output embedding layer.
        """
        return self.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embedding layer to new_embeddings.

        Args:
            new_embeddings (torch.nn.Module): New embedding layer.
        """
        self.lm_head.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        """Forward pass for the BertLMHeadModel.

        Args:
            input_ids (torch.Tensor, optional):
                See :class:`BertModel`.
            attention_mask (torch.Tensor, optional):
                See :class:`BertModel`.
            token_type_ids (torch.Tensor, optional):
                See :class:`BertModel`.
            position_ids (torch.Tensor, optional):
                See :class:`BertModel`.
            head_mask (torch.Tensor, optional):
                See :class:`BertModel`.
            labels (torch.Tensor, optional):
                Labels for computing the left-to-right language modeling loss.
            use_cache (bool, optional):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            output_attentions (bool, optional):
                Whether to return attentions tensors of all attention layers.
            output_hidden_states (bool, optional):
                Whether to return hidden states of all layers.
            return_dict (bool, optional):
                Whether to return a :class:`~CausalLMOutput` object.

        Returns:
            :class:`~CausalLMOutput`: An instance of CausalLMOutput if `return_dict=True`.
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)
        if labels is not None:
            use_cache = False

        outputs: BertModelOutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # Shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :
                                                          -1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores, )
            if output_hidden_states:
                output += outputs[2:]
            if output_attentions:
                output += outputs[3:]
            return (lm_loss, ) + output if lm_loss is not None else output

        return CausalLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      use_cache=True,
                                      **model_kwargs):
        """
        Prepare inputs for generation.
        Args:
            input_ids (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Attention mask.
            use_cache (bool, optional): Whether to use caching.

        Returns:
            dict: Prepared inputs for generation.
        """
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'use_cache': use_cache,
        }


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model for Masked Language Modeling.

    Args:
        config: BERT configuration object.
    """

    def __init__(self, config: BertConfig):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.lm_head = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        """Get the output embedding layer.

        Returns:
            nn.Module: Output embedding layer.
        """
        return self.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module):
        """Set the output embedding layer.

        Args:
            new_embeddings (nn.Module): New embedding layer.
        """
        self.lm_head.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """Perform forward pass for Masked Language Modeling.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            position_ids (torch.Tensor, optional): Positional embeddings.
            head_mask (torch.Tensor, optional): Head mask for attention layers.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary of outputs.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            Union[Tuple[torch.Tensor], MaskedLMOutput]: Either a tuple of output tensors or a MaskedLMOutput object.
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        outputs: BertModelOutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return (((masked_lm_loss, ) +
                     output) if masked_lm_loss is not None else output)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> dict:
        """Prepare inputs for generation.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            **model_kwargs: Additional model-specific arguments.

        Returns:
            dict: Prepared inputs for generation.
        """
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # Add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError('The PAD token should be defined for generation')

        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_zeros((attention_mask.shape[0], 1))
            ],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class BertForNextSentencePrediction(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if 'next_sentence_label' in kwargs:
            warnings.warn(
                'The `next_sentence_label` argument is deprecated and will be removed in a future version, use'
                ' `labels` instead.',
                FutureWarning,
            )
            labels = kwargs.pop('next_sentence_label')

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        outputs: BertModelOutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2),
                                          labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores, ) + outputs[2:]
            return (((next_sentence_loss, ) +
                     output) if next_sentence_loss is not None else output)

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        outputs: BertModelOutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long
                                              or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForMultipleChoice(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)
        num_choices = input_ids.shape[1]

        input_ids = (input_ids.view(-1, input_ids.size(-1))
                     if input_ids is not None else None)
        attention_mask = (attention_mask.view(-1, attention_mask.size(-1))
                          if attention_mask is not None else None)
        token_type_ids = (token_type_ids.view(-1, token_type_ids.size(-1))
                          if token_type_ids is not None else None)
        position_ids = (position_ids.view(-1, position_ids.size(-1))
                        if position_ids is not None else None)

        outputs: BertModelOutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss, ) +
                    output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
