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
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()

from enhancement import VisualDescriptionEnhancement
from eval_model import BaseEvalModel
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

from eval_datasets import *
from classification_utils import *



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

#parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=10)

parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--rices_type",
    default=None,
    help="Type to use RICES for generate description in this demo.",
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

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    print(f"Evaluating on {args.dataset_name} Dataset...")

    assert args.rices_type in ["image", "text", "ETD", None]
    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None:
        if args.rices_type == "ETD":
            cached_features = torch.load(
                f"{args.cached_demonstration_features}/image_{args.dataset_name}.pkl", map_location="cpu"
            )
    else:
        cached_features = None
    for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
        output = generate_description(
            args,
            eval_model=eval_model,
            batch_size=args.batch_size,
            seed=seed,
            dataset_name=args.dataset_name,
            cached_features=cached_features,
        )

    if args.rank == 0:
        results_file_path = os.path.join("results", f'{args.dataset_name}_description.txt')
        if args.results_file is not None:
                with open(results_file_path, 'w') as file:
                    for item in output:
                        file.write(f'{item}\n')


def generate_description(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    batch_size: int,
    seed: int = 42,
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
        prompt_fn = lambda x: eval_model.get_imagenet_description_prompt(label=x["class_name"])
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
        prompt_fn = lambda x: eval_model.get_cub_description_prompt(label=x["class_name"])
        all_class_names = CUB_CLASSNAMES
        k = 5
    elif dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        prompt_fn = lambda x: eval_model.get_car_description_prompt(label=x["class_name"])
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
        prompt_fn = lambda x: eval_model.get_dog_description_prompt(label=x["class_name"])
        all_class_names = STANFORD_DOG_CLASSNAMES
        k = 5
    elif dataset_name == "food101":
        train_dataset = Food101Dataset(
            root=args.dataset_root
        )
        test_dataset = Food101Dataset(
            root=args.dataset_root,
            train=False
        )
        prompt_fn = lambda x: eval_model.get_description_prompt(label=x["class_name"])
        all_class_names = FOOD101_NAMES
        k = 5
    elif dataset_name == "flowers102":
        train_dataset = Flowers102Dataset(
            root=args.dataset_root
        )
        test_dataset = Flowers102Dataset(
            root=args.dataset_root,
            train=False
        )
        prompt_fn = lambda x: eval_model.get_description_prompt(label=x["class_name"])
        all_class_names = FLOWERS102_NMAES
   
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    labels = all_class_names
    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))
    if args.rices_type == "ETD":
        print("description has been activated...")
        rices_etd = VisualDescriptionEnhancement(
            train_dataset,
            labels,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    
        
    num_batches = len(labels) // batch_size + (1 if len(labels) % batch_size != 0 else 0)
    label_batches = [labels[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    utils.random_seed(seed, args.rank)
    all_outputs = []
    cnt = 0
    for label_batch in tqdm(label_batches, desc="Processing batches", unit="batch"):
        # 为每个批次的标签找到图像数据
        batch_image_data = rices_etd.find_batch(label_batch)

        # 准备批处理的文本和图像
        batch_text = []
        batch_images = []
        for image_data in batch_image_data:
            if image_data is not None:
                context_text = prompt_fn(image_data)
                context_images = image_data['image']
                batch_text.append(context_text)
                batch_images.append([context_images])  

        # 如果批次中有图像数据，将其发送到模型
        if batch_text:
            if cnt == 0:
                print(batch_text[0])
                cnt += 1
            # 调用模型的方法来获取输出（请根据您的模型和需求修改此处）
            outputs = eval_model.get_outputs(
                batch_text=batch_text,
                batch_images=batch_images,
                min_generation_length=5,
                max_generation_length=17,
                num_beams=1,
                length_penalty=0.2,
                no_repeat_ngram_size=2,
                temperature=0.2,
                top_k=20,
            )
            all_outputs.extend(outputs)
        # ensemble logprobs together
    return all_outputs   

if __name__ == "__main__":
    main()
