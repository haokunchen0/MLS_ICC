from typing import List

from PIL import Image
import torch

from transformers import IdeficsForVisionText2Text, AutoProcessor, AutoTokenizer
from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype


class EvalModel(BaseEvalModel):
    """IDEFICS model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "cpu"
    """

    def __init__(self, model_args):
        assert (
            "lm_path" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "precision" in model_args
        ), "IDEFICS requires processor_path, lm_path, and device arguments to be specified"
        device_id = model_args["device"]
        self.device = (
            f"cuda:{device_id}"
            if ("device" in model_args)
            else "cpu"
        )

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_args["checkpoint_path"],
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            cache_dir=model_args["lm_path"],
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_args["checkpoint_path"],
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            cache_dir=model_args["lm_path"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_args["lm_tokenizer_path"])

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

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        encodings = self.processor.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            outputs = unwrap_model(self.model).generate(
                self._prepare_images(batch_images).to(self.device),
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_generation_length,
                min_new_tokens=min_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_vqa_prompt(self, question, answer=None) -> str:
        return (
            f"Question:{question} Short answer:{answer if answer is not None else ''}"
        )

    def get_caption_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"

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
    
    def get_imagenet_prompt(self, label=None) -> str:
        return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
    def get_imagenet_prompt_with_des(self, description, label=None) -> str:
        return f"<image>has {description}Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"
