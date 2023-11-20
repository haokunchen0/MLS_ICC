from typing import List, Dict

from PIL import Image
import torch
from einops import repeat

from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.src.factory import create_model_and_transforms
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype
from transformers.modeling_outputs import CausalLMOutputWithPast


    
class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args):
        assert (
            "vision_encoder_path" in model_args
            and "lm_path" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "cross_attn_every_n_layers" in model_args
            and "vision_encoder_pretrained" in model_args
            and "precision" in model_args
        ), "OpenFlamingo requires vision_encoder_path, lm_path, device, checkpoint_path, lm_tokenizer_path, cross_attn_every_n_layers, vision_encoder_pretrained, and precision arguments to be specified"

        self.device = (
            model_args["device"]
            if ("device" in model_args and model_args["device"] >= 0)
            else "cpu"
        )

        (
            self.model,
            self.image_processor,
            self.tokenizer,
        ) = create_model_and_transforms(
            model_args["vision_encoder_path"],
            model_args["vision_encoder_pretrained"],
            model_args["lm_path"],
            model_args["lm_tokenizer_path"],
            cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
        )
        checkpoint = torch.load(model_args["checkpoint_path"], map_location=self.device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"

        self.lm_name = model_args["lm_path"].split("/")[-1]

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(
                self.device, dtype=self.cast_dtype, non_blocking=True
            )
        return batch_images

    def _prepare_text(
        self,
        batch: List[List[str]],
        padding="longest",
        truncation=True,
        max_length=2000,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (B, T_txt)
            attention_mask (tensor)
                shape (B, T_txt)
        """
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        input_ids = input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True)
        attention_mask = attention_mask.to(
            self.device, dtype=self.cast_dtype, non_blocking=True
        )
        return input_ids, attention_mask.bool()

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """
        Get generation outputs.
        """
        batch_images = self._prepare_images(batch_images)
        input_ids, attention_mask = self._prepare_text(batch_text)

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    batch_images,
                    input_ids,
                    attention_mask,
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        normalize_length: bool=True,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        """
        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )
        _vision_x = batch_images
 
        outputs = self.__call__(
            vision_x=_vision_x,
            lang_x=_lang_x,
            attention_mask=_attention_mask,
        )

        classnames_tokens = self.tokenizer(
                all_class_names
            )["input_ids"]

        overall_log_probs = []
        first_token_log_probs = []
        batch_size = outputs.scores[0].shape[0]
        # overall_probs = []
        # first_toke_probs = []
        # for classname_tokens in classnames_tokens:
        #     classname_tokens_num = len(classname_tokens)
        #     prob = torch.ones(batch_size).to(self.device)
        #     for i in range(classname_tokens_num):
        #         try:
        #             scores = torch.softmax(outputs.scores[i],dim=-1)
        #             if i == 0:
        #                 first_token_prob = scores[:, classname_tokens[i]]
        #             prob *= scores[:, classname_tokens[i]]
        #         except IndexError as e:
        #             prob = torch.zeros(batch_size).to(self.device)
        #     overall_probs.append(prob) # (B, 1)
        #     first_token_probs.append(first_token_prob)
        
        # overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
        # first_token_probs = torch.vstack(first_token_probs).T.cpu()
        # return overall_probs
        for classname_tokens in classnames_tokens:
            classname_tokens_num = len(classname_tokens)
            log_prob = torch.zeros(batch_size).to(self.device)
            for i in range(classname_tokens_num):
                try:
                    # Compute log probabilities instead of probabilities
                    log_scores = torch.nn.functional.log_softmax(outputs.scores[i], dim=-1)
                    if i == 0:
                        first_token_log_prob = log_scores[:, classname_tokens[i]]
                    # Sum the log probabilities instead of multiplying probabilities
                    log_prob += log_scores[:, classname_tokens[i]]
                except IndexError as e:
                    log_prob = torch.full((batch_size,), -float('inf')).to(self.device)  # Use negative infinity to represent log(0)
            if normalize_length:
                log_prob /= classname_tokens_num
            overall_log_probs.append(log_prob)  # (B, 1)
            first_token_log_probs.append(first_token_log_prob)

        overall_log_probs = torch.vstack(overall_log_probs).T.cpu()  # shape (B, num_classes)
        first_token_log_probs = torch.vstack(first_token_log_probs).T.cpu()
        return overall_log_probs

    def __call__(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor,
        min_generation_length: int = 1,
        max_generation_length: int = 20,
        num_beams: int = 1,
        length_penalty: float = 1,
    ):
        """
        Calls the forward function of the model.
        Special logic to handle the case if past_key_values is not None:
            then lang_x is assumed to contain the tokens to be generated
            *excluding* the tokens already in past_key_values.
            We then repeatedly call forward, updating the past_key_values.
        """
        # standard forward pass
        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    vision_x,
                    lang_x,
                    attention_mask,
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            return outputs

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def get_imagenet_prompt(self, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
    def get_imagenet_prompt_with_des(self, description, label=None) -> str:
        return f"<image>has {description}Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
    def get_imagenet_prompt_with_des_new(self, description, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}, which has {description}{'<|endofchunk|>' if label is not None else ''}"
    def get_imagenet_prompt_with_des2(self, description, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}, has {description}{'<|endofchunk|>' if label is not None else ''}"