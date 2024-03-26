import argparse
import importlib
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import utils
import math

from rices import RICES_Image, RICES_Text
from enhancement import LabelDistributionEnhancement
from tqdm import tqdm
from eval_model import BaseEvalModel
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

from templates import *
from eval_datasets import *
from classification_utils import *
from class_description import *

import transformers
transformers.logging.set_verbosity_error()
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` and `Idefics` is supported.",
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

parser.add_argument("--batch_size_map", type=str, default="0:256,1:128,2:64,4:32,8:16,16:8")

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
parser.add_argument(
    "--label_cached_demonstration_features",
    default=None,
    help="for LDE image to label..."
)
parser.add_argument(
    "--method_type",
    default="SL",
    help="SL/LDE/VDE/ensemble."
)
parser.add_argument(
    "--LDE_type",
    default="EL",
    help="EL/DL/DD/(RL)/(RP)"
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

parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="device of GPUs.",
)

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

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    print(f"Evaluating {args.model} on {args.dataset_name} Dataset...")

    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None:
        cached_features = torch.load(
            f"{args.cached_demonstration_features}/image_{args.dataset_name}.pkl", map_location="cpu"
        )
    else:
        cached_features = None

    if args.method_type != "SL":
        label_cached_features = torch.load(
                f"/home/chk/data/icl/text_{args.dataset_name}_new.pkl", map_location="cpu"
            )
    else:
        label_cached_features = None    
        
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
    label_cached_features=None,
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
    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.dataset_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.dataset_root, "val"))
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = IMAGENET_CLASSNAMES
        description = IMAGENET_DESCRIPTION
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
        description = CUB200_DESCRIPTION
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
        description = STANFORD_CARS_DESCRIPTION
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
        description = STANFORD_DOG_DESCRIPTION
        k = 5
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")
    templates = OPENAI_IMAGENET_TEMPLATES 
    labels = all_class_names
    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
    )
    # Choose rices types
    if args.rices_type == "image":
        print("rices has been activated...")
        rices_image_dataset = RICES_Image(
            train_dataset,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)
        
    # Enhancement methods.
    if args.method_type == "LDE" or args.method_type == "ensemble":

        enhancement = LabelDistributionEnhancement(
            train_dataset,
            labels,
            templates,
            eval_model.device,
            batch_size,
            cached_features=label_cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
            LDE_type=args.LDE_type,
        )
        
    utils.random_seed(seed, args.rank)
    predictions = []
    cnt=0
    for _, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name},{num_shots}",
        total=len(test_dataloader),
    ):
        
        if args.rices_type == "image":
            batch_demo_samples = rices_image_dataset.find(batch["image"], effective_num_shots)
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
            vde_text, lde_text = [], []
            for i in range(len(batch["image"])):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])
                
                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])
                
                if args.method_type == "VDE" or args.method_type == "ensemble":
                    context_text = "".join([eval_model.get_imagenet_prompt_with_des(description[x["class_id"]], label=x["class_name"]) for x in batch_demo_samples[i]])

                    # Keep the text but remove the image tags for the zero-shot case
                    if num_shots == 0:
                        context_text = context_text.replace("<image>", "")

                    batch_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                    )
                    vde_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                    )

                if args.method_type == "LDE" or args.method_type == "ensemble":
                    true_labels_for_current_batch = [x['class_name'] for x in batch_demo_samples[i]]
                    if args.LDE_type == "RL":
                        similar_labels_for_batch = [[random.choice(all_class_names) for _ in true_labels_for_current_batch]]
                    else:
                        similar_labels_for_batch = enhancement.MultilabelwithSimilarity([x['image'] for x in batch_demo_samples[i]], true_labels_for_current_batch, 1)
                    
                    # Combine real tags with similar tags
                    combined_labels_for_batch = []
                    for true_label, similar_labels in zip(true_labels_for_current_batch, similar_labels_for_batch):
                        combined_labels = [true_label] + similar_labels
                        combined_labels_for_batch.append(combined_labels)

                    # Connect the tags of each image
                    labels_for_each_image = [",".join(labels) for labels in combined_labels_for_batch]

                    # Use the prompt_fn function to generate a prompt for each image.
                    prompts_for_each_image = [prompt_fn({"class_name": labels}) for labels in labels_for_each_image]
                    context_text = "".join(prompts_for_each_image)
                    # Keep the text but remove the image tags for the zero-shot case
                    if num_shots == 0:
                        context_text = context_text.replace("<image>", "")

                    batch_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                        )
                    lde_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                    )
                if args.method_type == "SL":
                    context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])
                    if num_shots == 0:
                        context_text = context_text.replace("<image>", "")
                        # Keep the text but remove the image tags for the zero-shot case
                    batch_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                        )
                if args.method_type == "RL":
                    modified_batch_demo_samples = [{**x, 'class_name': random.choice(all_class_names)} if 'class_name' in x else x for x in batch_demo_samples[i]]
                    context_text = "".join([prompt_fn(x) for x in modified_batch_demo_samples])
                    if num_shots == 0:
                        context_text = context_text.replace("<image>", "")
                        # Keep the text but remove the image tags for the zero-shot case
                    batch_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                        )
                    
            if cnt == 0:
                print("*"*20)
                print("Prompt Example:")
                print((vde_text[0], lde_text[0]) if args.method_type == "ensemble" else batch_text[0])
                cnt += 1
            # get predicted class names
            if args.method_type == "ensemble":
                logprobs.append(
                    eval_model.get_rank_classifications(
                        vde_text,
                        batch_images,
                        all_class_names,
                        normalize_length=True,
                    ) + eval_model.get_rank_classifications(
                        lde_text,
                        batch_images,
                        all_class_names,
                        normalize_length=True,
                    )
                )
            else:
                logprobs.append(
                    eval_model.get_rank_classifications(
                        batch_text,
                        batch_images,
                        all_class_names,
                        normalize_length=True,
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
                    "gt_id":batch['class_id'][i],
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
