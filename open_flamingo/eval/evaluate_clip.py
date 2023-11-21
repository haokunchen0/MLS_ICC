import argparse
import importlib
import json
import os

from collections import defaultdict

import numpy as np
import torch
import utils

from rices import RICES_Text
from tqdm import tqdm
from eval_model import BaseEvalModel
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

from eval_datasets import (
    ImageNetDataset,
    CUB200Dataset,
    StanfordCarDataset, 
    StanfordDogDataset,
)
from templates import *
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

parser.add_argument("--batch_size", type=int, default=50)

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


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()


    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    print(f"Evaluating on {args.dataset_name} Dataset...")

    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None:

        cached_features = torch.load(
                f"{args.cached_demonstration_features}/text_{args.dataset_name}_new.pkl", map_location="cpu"
        )
    else:
        cached_features = None

    device_id = eval_model.device
    del eval_model.model
    torch.cuda.empty_cache()

    batch_size = args.batch_size
    print(f"Now evaluating on {batch_size} samples...")
    scores = []
    for seed, _ in zip(args.trial_seeds, range(args.num_trials)):
        acc = evaluate_clip(
            args,
            batch_size=batch_size,
            seed=seed,
            device=device_id,
            dataset_name=args.dataset_name,
            cached_features=cached_features,
        )
        print(f"{args.dataset_name} Acc: {acc}")
            
        scores.append(acc)

def evaluate_clip(
    args: argparse.Namespace,
    batch_size: int,
    seed: int = 42,
    device: str='cpu',
    dataset_name: str = "imagenet",
    cached_features=None,
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
        all_class_names = IMAGENET_CLASSNAMES
    elif dataset_name == "cub200":
        train_dataset = CUB200Dataset(
            root=args.dataset_root
        )
        test_dataset = CUB200Dataset(
            root=args.dataset_root,
            train=False
        )
        all_class_names = CUB_CLASSNAMES
    elif dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        all_class_names = STANFORD_CAR_CLASSNAMES
    elif dataset_name == "stanford_dog":
        train_dataset = StanfordDogDataset(
            root=args.dataset_root
        )
        test_dataset = StanfordDogDataset(
            root=args.dataset_root,
            train=False
        )
        all_class_names = STANFORD_DOG_CLASSNAMES
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
    )
    templates = OPENAI_IMAGENET_TEMPLATES 

    print("rices_text has been activated...")
    rices_text_labels = RICES_Text(
            train_dataset,
            all_class_names,
            templates,
            device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )


    utils.random_seed(seed, args.rank)
    results = np.array([])
    # print(len(test_dataloader))
    sum_correct = 0
    for _, batch in tqdm(
        enumerate(test_dataloader),total=len(test_dataloader)
    ):       
        _,indices = rices_text_labels.find(batch["image"], 1)

        class_ids = np.array(batch['class_id'])
        pred_labels = np.array(indices)[:, 0]
        results = np.concatenate((results, pred_labels))
        correct = (class_ids == pred_labels).sum()
        sum_correct += correct

    return sum_correct / len(test_dataset)


if __name__ == "__main__":
    main()
