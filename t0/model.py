import logging
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, MODEL_FOR_CAUSAL_LM_MAPPING, \
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

logger = logging.getLogger(__name__)

class ModelBase(nn.Module):
    def forward(self, batch):
        raise NotImplementedError

    @staticmethod
    def from_config(config, **kwargs) -> "ModelBase":
        task_mapping = [
            (MODEL_FOR_CAUSAL_LM_MAPPING, DecoderModel),
            (MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, EncoderDecoderModel),
        ]
        config_name = config.__class__
        for transformer_model_mapping, model in task_mapping:
            transformer_model_name = transformer_model_mapping.get(config_name, None)
            if transformer_model_name is not None:
                return model(config=config, **kwargs)

        raise NotImplementedError

class EncoderDecoderModel(ModelBase):
    def __init__(self, config, model_name_or_path: Optional[str], parallelize: bool, **kwargs):
        """

        Args:
            config:
            model_name_or_path:
            parallelize:
            device: if parallelize = False, then we use specified device.
        """
        super(EncoderDecoderModel, self).__init__()
        logger.info("Building EncoderDecoderModel")
        if model_name_or_path:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            self._model = AutoModelForSeq2SeqLM.from_config(config)

        if parallelize:
            assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
            self._model.parallelize()


    def forward(self, batch) -> torch.Tensor:
        model_inputs = {
            k: batch[k]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        # print(model_inputs)
        # print(model_inputs['input_ids'].size()) # [16, 1024]
        # print(model_inputs['labels'].size()) # [16, 4]
        logits = self._model(**model_inputs).logits
        # print(logits.size()) # [16, 4 ,32128]
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        # print(masked_log_probs.size()) # [16, 4 ,32128]
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        # print(seq_token_log_probs.size()) # [16, 4 ,1]
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        # print(seq_log_prob.size()) # [16]
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),
                                         -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        # print(seq_log_prob.size()) # [8, 2]
        predictions = seq_log_prob.argmax(dim=-1)
        # print(predictions)
        return predictions

class DecoderModel(ModelBase):
    def __init__(self, config, model_name_or_path: Optional[str], **kwargs):
        super(DecoderModel, self).__init__()
        logger.info("Building DecoderModel")
        if model_name_or_path:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            self._model = AutoModelForCausalLM.from_config(config)

    def forward(self, batch):
        _, prefix_length = batch["input_ids"].shape
        model_inputs = {
            "input_ids": torch.cat([batch["input_ids"], batch["labels"]], dim=-1),
            "attention_mask": torch.cat([batch["attention_mask"], batch["labels_attention_mask"]], dim=-1),
        }
        # Set position ids correctly to take care of padding tokens between inputs_ids and labels
        # Empty attention_mask is a forbidden value, ie full of zeros. In fact the first element should be 1 as the input
        #   cannot be empty
        assert torch.all(model_inputs["attention_mask"][:,0] == 1), "First element in the attention mask should be 1."
        position_ids = torch.cumsum(model_inputs["attention_mask"].to(torch.long), dim=-1) - 1
        model_inputs["position_ids"] = position_ids

        logits = self._model(**model_inputs).logits[:, prefix_length-1:-1]
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),
                                         -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        predictions = seq_log_prob.argmax(dim=-1)
        return predictions



### return confidence ###

class ModelBase_with_confidence(nn.Module):
    def forward(self, batch):
        raise NotImplementedError

    @staticmethod
    def from_config(config, **kwargs) -> "ModelBase_with_confidence":
        task_mapping = [
            (MODEL_FOR_CAUSAL_LM_MAPPING, DecoderModel),
            (MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, EncoderDecoderModel_with_confidence),
        ]
        config_name = config.__class__
        for transformer_model_mapping, model in task_mapping:
            transformer_model_name = transformer_model_mapping.get(config_name, None)
            if transformer_model_name is not None:
                return model(config=config, **kwargs)

        raise NotImplementedError

class EncoderDecoderModel_with_confidence(ModelBase_with_confidence):
    def __init__(self, config, model_name_or_path: Optional[str], parallelize: bool, **kwargs):
        """

        Args:
            config:
            model_name_or_path:
            parallelize:
            device: if parallelize = False, then we use specified device.
        """
        super(EncoderDecoderModel_with_confidence, self).__init__()
        logger.info("Building EncoderDecoderModel")
        if model_name_or_path:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            self._model = AutoModelForSeq2SeqLM.from_config(config)

        if parallelize:
            assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
            self._model.parallelize()


    def forward(self, batch) -> torch.Tensor:
        model_inputs = {
            k: batch[k]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        # print(model_inputs)
        # print(model_inputs['input_ids'].size()) # [16, 1024]
        # print(model_inputs['labels'].size()) # [16, 4]
        logits = self._model(**model_inputs).logits
        # print(logits.size()) # [16, 4 ,32128]
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        # print(masked_log_probs.size()) # [16, 4 ,32128]
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        # print(seq_token_log_probs.size()) # [16, 4 ,1]
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        # print(seq_log_prob.size()) # [16]
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),
                                         -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        print(seq_log_prob.size()) # [8, 2]
        max_ = torch.max(seq_log_prob, dim = -1)
        print(max_)
        min_ = torch.min(seq_log_prob, dim = -1)
        print(min_)
        confidence = torch.div(max_, min_)
        print(confidence)
        quit()

        predictions = seq_log_prob.argmax(dim=-1)
        # print(predictions)
        return predictions, confidence
