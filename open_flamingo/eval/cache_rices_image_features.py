'''
Cache image features for test samples
'''
import argparse
import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
    )
)
from rices import RICES_Image
from eval_datasets import *
import os
import torch
from classification_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the cached features.",
)
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--batch_size", default=4,type=int)


parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="idx of GPUs."
)


## Imagenet dataset
parser.add_argument("--dataset_root", type=str, default="/data")

def main():
    args, _ = parser.parse_known_args()
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    # cache image features for test samples
    print("Caching ...")
    train_dataset = CUB200Dataset(root=args.dataset_root)
    rices_dataset = RICES_Image(
        dataset=train_dataset,
        device=args.device,
        batch_size=args.batch_size,
        vision_encoder_path=args.vision_encoder_path,
        vision_encoder_pretrained=args.vision_encoder_pretrained
    )
    torch.save(
        rices_dataset.features,
        os.path.join(args.output_dir, "image_cub200_new.pkl"),
    )



if __name__ == "__main__":
    main()
 