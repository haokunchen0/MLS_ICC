import argparse
import importlib
import json
import os
import uuid
import random
from collections import defaultdict

import numpy as np
import torch
import utils
import math

from rices import RICES_Image, RICES_Text, RICES_Both
from tqdm import tqdm
from eval_model import BaseEvalModel
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

from eval_datasets import (
    ImageNetDataset,
    CUB200Dataset,
    StanfordCarDataset, 
    StanfordDogDataset,
)
import transformers
transformers.logging.set_verbosity_error()
from classification_utils import (
    IMAGENET_CLASSNAMES,
    CUB_CLASSNAMES,
    STANFORD_CAR_CLASSNAMES,
    STANFORD_DOG_CLASSNAMES,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 1, 2, 4, 8], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

#parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--batch_size_map", type=str, default="0:40,1:32,2:24,4:16,8:4,16:2,32:1")

parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--rices_type",
    default=None,
    help="Type to use RICES for evaluation, image or text. If none, uses random demonstrations.",
)
parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default=None,
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)

# Dataset arguments
parser.add_argument("--dataset_name", type=str, default="imagenet")
parser.add_argument("--dataset_root", type=str, default="/tmp")

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)

def parse_batch_size_map(batch_size_map_str):
    mapping = {}
    pairs = batch_size_map_str.split(',')
    for pair in pairs:
        shot, batch_size = pair.split(':')
        mapping[int(shot)] = int(batch_size)
    return mapping

def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
    args.batch_size_map = parse_batch_size_map(args.batch_size_map)

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    print(f"Evaluating on {args.dataset_name} Dataset...")

    assert args.rices_type in ["image", "text", "both", None]
    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None and args.rices_type is not None:
        if args.rices_type == "both":
            cached_features = [None, None]
            cached_features[0] = torch.load(
                f"{args.cached_demonstration_features}/image_{args.dataset_name}.pkl", map_location="cpu"
            )
            cached_features[1] = torch.load(
                f"{args.cached_demonstration_features}/text_{args.dataset_name}.pkl", map_location="cpu"
            )
        else:
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/{args.rices_type}_{args.dataset_name}.pkl", map_location="cpu"
            )
    else:
        cached_features = None

    for shot in args.shots:
        results = {f"{args.dataset_name}": []}
        batch_size = args.batch_size_map.get(shot, args.batch_size_map[0])
        print(f"Now evaluating on {batch_size} samples...")
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            classification_score = evaluate_classification(
                args,
                eval_model=eval_model,
                num_shots=shot,
                batch_size=batch_size,
                seed=seed,
                dataset_name=args.dataset_name,
                cached_features=cached_features,
                label_cached_features=label_cached_features,
                use_prompt_ensembling=args.classification_prompt_ensembling,
            )
            if args.rank == 0:
                print(
                    f"Shots {shot} Trial {trial} "
                    f"{args.dataset_name} score: {classification_score}"
                )
                scores.append(classification_score)

        if args.rank == 0:
            print(f"Shots {shot} Mean {args.dataset_name} score: {np.nanmean(scores)}")
            results[f"{args.dataset_name}"].append(
                {
                    "shots": shot,
                    "acc": scores,
                }
            )
            if args.results_file is not None:
                with open(args.results_file, "a") as f:
                    json.dump(results, f)
                    f.write('\n')


def evaluate_classification(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    batch_size: int,
    seed: int = 42,
    num_shots: int = 8,
    dataset_name: str = "imagenet",
    cached_features=None,
    use_prompt_ensembling: bool = False,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo"
        )

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.dataset_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.dataset_root, "val"))
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = IMAGENET_CLASSNAMES
        k = 5
    elif dataset_name == "cub200":
        train_dataset = CUB200Dataset(
            root=args.dataset_root
        )
        test_dataset = CUB200Dataset(
            root=args.dataset_root,
            train=False
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = CUB_CLASSNAMES
        k = 5
    elif dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = STANFORD_CAR_CLASSNAMES
        k = 5
    elif dataset_name == "stanford_dog":
        train_dataset = StanfordDogDataset(
            root=args.dataset_root
        )
        test_dataset = StanfordDogDataset(
            root=args.dataset_root,
            train=False
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = STANFORD_DOG_CLASSNAMES
        k = 5
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    labels = all_class_names
    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
    )

    if args.rices_type == "both":
        rices_both_dataset = RICES_Both(
            train_dataset,
            labels,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )

    elif args.rices_type == "image":
        rices_image_dataset = RICES_Image(
            train_dataset,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    elif args.rices_type == "text":
        print("rices_text has been activated...")
        rices_text_labels = RICES_Text(
            train_dataset,
            labels,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    utils.random_seed(seed, args.rank)
    predictions = []
    for batch_idx, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name}",
        total=len(test_dataloader),
    ):
        if args.rices_type == "both":
            batch_demo_samples = rices_both_dataset.find(batch["image"], effective_num_shots)
        elif args.rices_type == "image":
            batch_demo_samples, batch_similarity_probs = rices_image_dataset.find(batch["image"], effective_num_shots)
        elif args.rices_type == "text":
            batch_demo_labels = rices_text_labels.find(batch["image"], 1)
            batch_demo_samples = rices_text_labels.get_images_from_labels(batch_demo_labels, effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )
        # set up prompt ensembling
        num_permutations = (
            min(6, math.factorial(effective_num_shots)) if use_prompt_ensembling else 1
        )
        logprobs = []
        for _ in range(num_permutations):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])

                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text
                    + prompt_fn({"class_name": None})
                )

            # get predicted class names
            logprobs.append(
                eval_model.get_rank_classifications(
                    batch_text,
                    batch_images,
                    all_class_names,
                )
            )

        # ensemble logprobs together
        logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1)

        predicted_classnames, predicted_logprobs = utils.get_predicted_classnames(
            logprobs,
            k,
            class_id_to_name,
        )

        # compute accuracy
        for i, topk in enumerate(predicted_classnames):
            y_i = batch["class_name"][i]
            score = torch.exp(
                predicted_logprobs[i][0] - torch.logsumexp(logprobs[i], dim=0)
            ).item()
            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": topk[0],
                    "pred_score": score,
                }
            )

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # return top-1 accuracy
    acc1 = sum(
        int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
    )
    return float(acc1) / len(all_predictions)


if __name__ == "__main__":
    main()
