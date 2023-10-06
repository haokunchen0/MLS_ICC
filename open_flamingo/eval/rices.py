import open_clip
import torch
from tqdm import tqdm
import torch
from utils import custom_collate_fn
import random

class RICES_Image:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []
        print(len(self.dataset))
        # Switch to evaluation mode
        self.model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                batch = batch["image"]
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch]
                ).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach())


        features = torch.cat(features)
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]


class RICES_Text:
    def __init__(
        self,
        dataset,
        text_label,
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
    ):
        self.dataset = dataset
        self.text_label = text_label
        self.device = device
        self.batch_size = batch_size

        # Load the model and tokenizer
        self.model, _, self.image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(vision_encoder_path)

        # Tokenize the text descriptions and move to device
        self.text_tokens = self.tokenizer(text_label).to(self.device)

        # Precompute features
        if cached_features is None:
            self.text_features = self._precompute_text_features()
        else:
            self.text_features = cached_features

    def _precompute_text_features(self):
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            text_features = self.model.encode_text(self.text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu()

    def find(self, batch, num_examples):
        """
        Get the ranked text descriptions based on similarity to the image.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Preprocess and encode the image
            image_input = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )
            #Get the feature of the input image
            query_feature = self.model.encode_image(image_input)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)
            # Compute the similarity between image and text representations
            similarities = (query_feature @ self.text_features.T).squeeze()
            if similarities.ndim == 1:
                similarities = similarities.unsqueeze(0)
            
            # Get the indices of the 'num_examples' most similar texts
            indices = similarities.argsort(dim=-1, descending=True)[:, :num_examples]
        # Return the most similar text descriptions
        return [[self.text_label[i] for i in row] for row in indices]
    
    def get_images_from_labels(self, batch_labels, shot):
        """
        Get 'shot' number of images for each label in the batch from the training dataset.
        """
        # Access dataset attributes based on dataset type
        if hasattr(self.dataset, 'image_paths') and hasattr(self.dataset, 'labels'):
            image_data = list(zip(self.dataset.image_paths, self.dataset.labels))
        elif hasattr(self.dataset, 'imgs'):
            image_data = self.dataset.imgs
        else:
            raise ValueError("Dataset does not have required attributes.")

        print(f"Using dataset: {type(self.dataset).__name__}")
        if hasattr(self.dataset, 'class_id_to_name'):
            print("Dataset has class_id_to_name attribute.")
        else:
            print("Dataset lacks class_id_to_name attribute.")

        results = []
        for label_list in batch_labels:
            label = label_list[0]  # As per your example, there's one label per batch item
            matching_indices = [
                idx for idx, (image, img_label) in enumerate(image_data)
                if self.dataset.class_id_to_name[img_label] == label
            ]
            random.seed(42)
            selected_indices = random.sample(matching_indices, shot)
            selected_samples = [self.dataset[idx] for idx in selected_indices]
            results.append(selected_samples)
        return results







