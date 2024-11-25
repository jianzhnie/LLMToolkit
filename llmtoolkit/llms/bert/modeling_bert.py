import logging
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import BertPreTrainedModel

from ..modeling_outputs import BertEncoderOutput, BertModelOutput
from .config_bert import BertConfig

logger = logging.getLogger(__name__)


def gelu(x):
    return nn.functional.gelu(x)


def swish(x):
    return x * nn.functional.sigmoid(x)


# torch.nn.functional.gelu(x) # Breaks ONNX export
ACT2FN = {
    'gelu': gelu,
    'tanh': nn.functional.tanh,
    'relu': nn.functional.relu,
    'swish': swish,
}


class BertEmbedding(nn.Module):
    """BERT embedding module that combines word, position, and token type
    embeddings.

    Args:
        config (BertConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size,
                                           config.hidden_size,
                                           padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,
                                               config.hidden_size)
        self.token_type_embedding = nn.Embedding(config.type_vocab_size,
                                                 config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the BERT embedding module.

        Args:
            input_ids (Optional[Tensor]): Input token indices.
            token_type_ids (Optional[Tensor]): Token type indices.
            position_ids (Optional[Tensor]): Position indices.

        Returns:
            Tensor: Combined word, position, and token type embeddings.
        """
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length,
                                        dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=input_ids.device)

        input_embeddings = self.word_embedding(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = input_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT Self Attention module.

    Args:
        config (BertConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """Transpose the dimensions of the input tensor for computing attention
        scores.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transposed tensor.
        """
        # (batch_size, seq_len, hidden_size) ->
        # (batch_size, seq_len, num_attention_heads, attention_head_size)
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        # (batch_size, num_attention_heads, seq_len, attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of the BERT self-attention module.

        Args:
            hidden_states (Tensor): Input hidden states.
            attention_mask (Optional[Tensor]): Attention mask tensor.
            head_mask (Optional[Tensor]): Head mask tensor.
            output_attentions (Optional[bool]): Whether to output attention scores.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Contextualized hidden states and optional attention scores.
        """

        # shape: (batch_size, seq_len, num_attention_heads * attention_head_size)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # shape: (batch_size, num_attention_heads, seq_len, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # (batch_size, num_attention_heads, seq_len, seq_len) *
        #           (batch_size, num_attention_heads, seq_len, attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, num_attention_heads, seq_len, attention_head_size) =>
        #            (batch_size,  seq_len, num_attention_heads, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # (batch_size, seq_len, all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = ((context_layer, attention_probs) if output_attentions else
                   (context_layer, ))
        return outputs


class BertSelfOutput(nn.Module):
    """BERT Self Output module. Processes the output from the self-attention
    layer.

    Args:
        hidden_size (int): Dimensionality of the input and output tensors.
        layer_norm_eps (float): Epsilon value for layer normalization.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
    """

    def __init__(self, config: BertConfig):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of the BERT Self Output module.

        Args:
            hidden_states (Tensor): Input hidden states from the previous layer.
            input_tensor (Tensor): The original input tensor to the sub-layer.

        Returns:
            Tensor: Processed hidden states after self-attention and feed-forward layers.
        """
        # Linear transformation
        hidden_states = self.dense(hidden_states)
        # Dropout for regularization
        hidden_states = self.dropout(hidden_states)
        # Residual connection and layer normalization
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT Attention module. Combines self-attention and output
    transformations.

    Args:
        hidden_size (int): Dimensionality of the input and output tensors.
        num_attention_heads (int): Number of attention heads.
        layer_norm_eps (float): Epsilon value for layer normalization.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
    """

    def __init__(self, config: BertConfig):
        super(BertAttention, self).__init__()
        # Self-attention layer
        self.self_atten = BertSelfAttention(config)
        # Output layer after self-attention
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass of the BERT Attention module.

        Args:
            hidden_states (Tensor): Input hidden states from the previous layer.
            attention_mask (Optional[Tensor]): Attention mask tensor.
            head_mask (Optional[Tensor]): Head mask tensor.
            output_attentions (Optional[bool]): Whether to output attention scores.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Processed hidden states and optional attention scores.
        """
        # Calculate self-attention and output
        self_outputs = self.self_atten(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # Include attention scores in the output if required
        outputs = (attention_output, ) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    """BERT Intermediate module. Processes the hidden states from the encoder
    layer.

    Args:
        hidden_size (int): Dimensionality of the hidden states.
        intermediate_size (int): Dimensionality of the intermediate states.
        hidden_act (Union[str, nn.Module]): Activation function applied to the intermediate hidden states.
    """

    def __init__(self, config: BertConfig):
        super(BertIntermediate, self).__init__()
        # Linear transformation layer
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # Activation function
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[
                config.hidden_act]  # Assume ACT2FN is a predefined dictionary
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of the BERT Intermediate module.

        Args:
            hidden_states (Tensor): Input hidden states from the encoder layer.

        Returns:
            Tensor: Processed intermediate hidden states.
        """
        # Linear transformation
        hidden_states = self.dense(hidden_states)
        # Activation function applied to intermediate states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT Output module. Combines linear transformation, dropout, and layer
    normalization.

    Args:
        hidden_size (int): Dimensionality of the input and output tensors.
        intermediate_size (int): Dimensionality of the intermediate tensors.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        layer_norm_eps (float): Epsilon value for layer normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """Forward pass of the BERT Output module.

        Args:
            hidden_states (Tensor): Input hidden states from the previous layer.
            input_tensor (Tensor): The original input tensor to the sub-layer.

        Returns:
            Tensor: Processed hidden states after linear transformation, dropout, and layer normalization.
        """
        # Linear transformation
        hidden_states = self.dense(hidden_states)
        # Dropout for regularization
        hidden_states = self.dropout(hidden_states)
        # Residual connection and layer normalization
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """BERT Layer module. Comprises attention, intermediate, and output sub-
    layers.

    Args:
        config (BertConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertLayer, self).__init__()
        # Self-attention layer
        self.attention = BertAttention(config)
        # Intermediate layer for further processing
        self.intermediate = BertIntermediate(config)
        # Output layer after intermediate processing
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, ...]:
        """Forward pass of the BERT Layer module.

        Args:
            hidden_states (Tensor): Input hidden states from the previous layer.
            attention_mask (Optional[Tensor]): Attention mask tensor.
            head_mask (Optional[Tensor]): Head mask tensor.
            output_attentions (Optional[bool]): Whether to output attention scores.

        Returns:
            Tuple[Tensor, ...]: Processed hidden states and optional attention scores.
        """
        # Calculate self-attention
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # Intermediate processing
        intermediate_output = self.intermediate(attention_output)

        # Output layer processing
        layer_output = self.output(intermediate_output, attention_output)

        # Include processed hidden states and optionally attention scores in the output
        outputs = (layer_output, ) + outputs
        return outputs


class BertEncoder(nn.Module):
    """BERT Encoder module. Stacks multiple layers of BERT layers.

    Args:
        config (BertConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertEncoder, self).__init__()
        # Stack multiple BERT layers
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[Tensor], BertEncoderOutput]:
        """Forward pass of the BERT Encoder module.

        Args:
            hidden_states (Tensor): Input hidden states from the previous layer.
            attention_mask (Optional[Tensor]): Attention mask tensor.
            head_mask (Optional[Tensor]): Head mask tensor.
            use_cache (Optional[bool]): Whether to use cached values for efficient computation.
            output_attentions (Optional[bool]): Whether to output attention scores.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return results as a dictionary.

        Returns:
            Union[Tuple[Tensor], BertEncoderOutput]: Processed hidden states and optional additional outputs.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                # Custom forward function for gradient checkpointing
                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                # Apply gradient checkpointing to the layer forward pass
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                # Standard layer forward pass
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    layer_outputs[1], )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None)
        return BertEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BertPooler(nn.Module):
    """Pool the result of BertEncoder by selecting the hidden state of the
    first token ([CLS] token) and passing it through a linear layer followed by
    Tanh activation."""

    def __init__(self, config: BertConfig):
        """Initialize the BertPooler.

        Args:
            config (BertConfig): Configuration object containing model parameters.
        """
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of the BertPooler.

        Args:
            hidden_states (Tensor): Hidden states from the BertEncoder.

        Returns:
            Tensor: Pooled output tensor.
        """
        # Select the hidden state of the first token ([CLS] token)
        first_token_tensor = hidden_states[:, 0]
        # Pass it through a linear layer and Tanh activation
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """Transformation module for the final prediction head in BERT models."""

    def __init__(self, config: BertConfig):
        """Initialize the BertPredictionHeadTransform.

        Args:
            hidden_size (int): Size of the hidden states.
            hidden_act (str or function): Activation function for the transformation.
            layer_norm_eps (float): Epsilon value for layer normalization.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Determine the activation function based on input type (string or function)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of the BertPredictionHeadTransform.

        Args:
            hidden_states (Tensor): Hidden states to be transformed.

        Returns:
            Tensor: Transformed hidden states.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """Prediction head for masked language modeling in BERT models."""

    def __init__(self, config: BertConfig):
        """Initialize the BertLMPredictionHead.

        Args:
            hidden_size (int): Size of the hidden states.
            vocab_size (int): Size of the vocabulary (number of tokens).
            hidden_act (str or function): Activation function for the transformation.
            layer_norm_eps (float): Epsilon value for layer normalization.
        """
        super(BertLMPredictionHead, self).__init__()
        # Transformation module for the hidden states
        self.transform = BertPredictionHeadTransform(config)
        # Decoder linear layer (output weights, no bias)
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # Output-only bias for each token
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Link the bias to the decoder so that it's resized correctly with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of the BertLMPredictionHead.

        Args:
            hidden_states (Tensor): Transformed hidden states.

        Returns:
            Tensor: Predicted logits for each token in the vocabulary.
        """
        # Transform the hidden states
        hidden_states = self.transform(hidden_states)
        # Predict logits for each token
        logits = self.decoder(hidden_states)
        return logits


class BertOnlyMLMHead(nn.Module):
    """MLM head for masked language modeling in BERT models."""

    def __init__(self, config: BertConfig):
        """Initialize the BertOnlyMLMHead.

        Args:
            config (BertConfig): Configuration object containing model parameters.
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass of the BertOnlyMLMHead.

        Args:
            hidden_states (Tensor): Transformed hidden states.

        Returns:
            Tensor: Predicted logits for each token in the vocabulary.
        """
        # Transform the hidden states
        # Predict logits for each token
        prediction_scores = self.predictions(hidden_states)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    """NSP (Next Sentence Prediction) head for BERT models."""

    def __init__(self, config: BertConfig):
        """Initialize the BertOnlyNSPHead.

        Args:
            config (BertConfig): Configuration object containing model parameters.
        """
        super(BertOnlyNSPHead, self).__init__()
        # Linear layer for predicting NSP relationship (binary classification)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:
        """Forward pass of the BertOnlyNSPHead.

        Args:
            pooled_output (Tensor): Pooled output from the BERT model.

        Returns:
            Tensor: Predicted scores for NSP relationship (binary classification).
        """
        # Predict NSP relationship scores
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    """Pre-training heads module for BERT models, including MLM and NSP
    heads."""

    def __init__(self, config: BertConfig):
        """Initialize the BertPreTrainingHeads.

        Args:
            config (BertConfig): Configuration object containing model parameters.
        """
        super(BertPreTrainingHeads, self).__init__()
        # Masked Language Modeling (MLM) head
        self.lm_head = BertOnlyMLMHead(config)
        # Next Sentence Prediction (NSP) head
        self.nsp_head = BertOnlyNSPHead(config)

    def forward(self, sequence_output: Tensor,
                pooled_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass of the BertPreTrainingHeads.

        Args:
            sequence_output (Tensor): Output from the BERT encoder for each token.
            pooled_output (Tensor): Pooled output from the BERT model.

        Returns:
            Tuple[Tensor, Tensor]: Predicted scores for MLM (word predictions) and NSP (binary classification).
        """
        # Predict masked tokens using MLM head
        prediction_scores = self.lm_head(sequence_output)
        # Predict next sentence relationship using NSP head
        seq_relationship_score = self.nsp_head(pooled_output)
        return prediction_scores, seq_relationship_score


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a
    Transformer").

    Args:
        config (BertConfig): A BertConfig class instance with the configuration to build a new model.
        add_pooling_layer (bool, optional): Whether to add a pooling layer. Defaults to True.

    Inputs:
        - input_ids (torch.Tensor): a torch.LongTensor of shape [batch_size, sequence_length],
            Indices of input sequence tokens in the vocabulary.
        - attention_mask (torch.Tensor, optional): an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        - token_type_ids (torch.Tensor, optional): an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        - position_ids (torch.Tensor, optional): Indices of positions of each input sequence token in the position embeddings.
        - head_mask (torch.Tensor, optional): Mask to control which heads are used in the attention layers.
        - use_cache (bool, optional): Whether to use caching for the computation. Defaults to None.
        - output_hidden_states (bool, optional): Whether to return the hidden states of all layers.
        - output_attentions (bool, optional): Whether to return the attentions tensors of all attention layers.
        - return_dict (bool, optional): Whether to return a ModelOutput object. Defaults to None.

    Outputs:
        BertModelOutput: A BertModelOutput containing various fields.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embedding

    def set_input_embeddings(self, value):
        self.embeddings.word_embedding = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertEncoderOutput]:
        r"""
        The BertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~torch.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`~torch.transformers.model_outputs.BertModelOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~torch.transformers.model_outputs.BertModelOutput`.
        """
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)),
                                        device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs: BertEncoderOutput = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (self.pooler(sequence_output)
                         if self.pooler is not None else None)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BertModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
