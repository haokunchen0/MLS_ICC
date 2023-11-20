import open_clip
import torch
from tqdm import tqdm
import torch
from utils import custom_collate_fn
import random
from open_clip import tokenizer
from functools import partial
from itertools import islice
from typing import Sequence, Callable, Union

def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch
'''
RICES_Image aims to retrieve the similiar images for test image from the training_set.
'''
class RICES_Image:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
        label_distribution=False
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.label_distribution = label_distribution
        
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

            similarity_probs = torch.nn.functional.softmax(similarity, dim=-1)
            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

            # Extract similarity values for the top indices
            top_similarity_probs = torch.gather(similarity_probs, 1, indices)
            
        similar_images = [[self.dataset[i] for i in row] for row in indices]
        similar_probs = [[prob.item() for prob in prob_row] for prob_row in top_similarity_probs]

        return similar_images
        # # Return with the most similar images last
        # return [[self.dataset[i] for i in reversed(row)] for row in indices]

'''
The RICES_Text is designed to retrieve images most similar to a given test image 
by first identifying similar labels and then sourcing images based on these labels.
'''
class RICES_Text:
    def __init__(
        self,
        dataset,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
        label_distribution=False
    ):
        self.dataset = dataset
        self.classnames = classnames
        self.templates = templates
        self.device = device
        self.batch_size = batch_size
        self.label_distribution = label_distribution
        
        # Load the model and tokenizer
        self.model, _, self.image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(vision_encoder_path)

        # Tokenize the text descriptions and move to device
        # self.text_tokens = self.tokenizer(classnames).to(self.device)
        # self.text_tokens = tokenizer.tokenize(self.text_label).to(self.device)
        # Precompute features
        if cached_features is None:
            self.text_features = self._precompute_text_features()
        else:
            # self.text_features = cached_features
            self.text_features = cached_features
        
    def _precompute_text_features(self):
        num_classes = len(self.classnames)
        import tqdm
        num_iter = 1 if self.batch_size is None else ((num_classes - 1) // self.batch_size + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=self.batch_size)
        
        batched_embeds = [self._process_batch(batch) for batch in iter_wrap(batched(self.classnames, self.batch_size))]
        zeroshot_weights = torch.cat(batched_embeds, dim=1)
        
        return zeroshot_weights.detach().cpu()
    
    def _process_batch(self, batch_classnames):
        use_format = isinstance(self.templates[0], str)
        num_templates = len(self.templates)
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in self.templates]
        texts = self.tokenizer(texts).to(self.device)
        self.model.eval()
        with torch.no_grad():
            class_embeddings = self.model.encode_text(texts, normalize=True)
            class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
            class_embeddings = class_embeddings.T
       
        return class_embeddings


    def find(self, batch, num_examples):
        """
        Get the ranked text descriptions based on similarity to the image.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Preprocess and encode the image
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )
            # batch = torch.Tensor(batch).to(device=self.device, dtype="fp32")
            # output = self.model(image=batch)

            #Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()
            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # image_features = inputs['image_features'] 
            # image_features = image_features.detach().cpu()

            # Compute the similarity between image and text representations
            similarity = (query_feature @ self.text_features).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            similarity_probs = torch.nn.functional.softmax(similarity, dim=-1)
            # similarities = 100. * image_features @ self.text_features
            
            # Get the indices of the 'num_examples' most similar texts
            indices = similarity_probs.argsort(dim=-1, descending=True)[:, :num_examples]
        # Return the most similar text descriptions
        return [[self.classnames[i] for i in row] for row in indices], indices
    
    
    def find_no_repeat(self, batch, true_labels, num_examples):
        """
        Get the ranked text descriptions based on similarity to the image but 
        avoid returning labels that match the true_labels of the images.
        """
        self.model.eval()

        with torch.no_grad():
            # Preprocess and encode the image
            image_input = torch.stack([self.image_processor(image) for image in batch]).to(self.device)
            
            # Get the feature of the input image
            query_feature = self.model.encode_image(image_input)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            # Compute the similarity between image and text representations
            similarities = (query_feature @ self.text_features.T)

            # Get the indices of the 'num_examples' most similar texts
            sorted_indices = similarities.argsort(dim=-1, descending=True)
            
            final_indices = []
            final_probs = []
            for idx, true_label in enumerate(true_labels):
                # Get indices of the top 'num_examples' similar labels excluding the true label
                top_similar_indices = [i for i in sorted_indices[idx] if self.text_label[i] != true_label][:num_examples]
                
                # Include the index of the true label
                true_label_index = self.text_label.index(true_label)
                similar_indices_with_true_label = [true_label_index] + top_similar_indices

                # Extract the original similarity values for these indices
                combined_similarities = similarities[idx, similar_indices_with_true_label]
                
                # Normalize the combined similarities to get the percentages
                percentages_with_true_label = combined_similarities / combined_similarities.sum()
                
                # Exclude the true label's percentage
                percentages = percentages_with_true_label[1:]
                final_probs.append(percentages)
                final_indices.append(top_similar_indices)

        if self.label_distribution:
            combined_texts = []
            for label_row, prob_row in zip(final_indices, final_probs):
                combined = [f" but maybe { prob*100:.2f} probability is {self.text_label[i]}" for i, prob in zip(label_row, prob_row)]
                combined_texts.append(combined)
            return combined_texts
        else:
            similar_texts = [[self.text_label[i] for i in row] for row in final_indices]
            return similar_texts
        
    def find_similar_labels(self, query_texts, num_similar=1):
        """
        For the provided list of query labels, find the most similar labels.
        """
        # 切换到评估模式
        self.model.eval()

        with torch.no_grad():
            # 对查询文本进行分词并移至设备
            query_tokens = self.tokenizer(query_texts).to(self.device)
            query_features = self.model.encode_text(query_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_features = query_features.detach().cpu()

            # 计算查询文本和所有标签之间的相似性
            similarities = (query_features @ self.text_features.T).squeeze()
            if len(similarities.shape) == 1:  # 处理单个查询文本的情况
                similarities = similarities.unsqueeze(0)
            # 获取最相似的标签的索引
            sorted_indices = similarities.argsort(dim=-1, descending=True)
            
        # 确保返回的标签与查询文本不同
        final_labels = []
        for idx, query_text in enumerate(query_texts):
            # Get indices of the top 'num_similar' similar labels excluding the true label
            top_similar_indices = [i for i in sorted_indices[idx] if self.text_label[i] != query_text][:num_similar]

            # Include the index of the true label
            true_label_index = self.text_label.index(query_text)
            similar_indices_with_true_label = [true_label_index] + top_similar_indices

            # Extract the original similarity values for these indices
            combined_similarities = similarities[idx, similar_indices_with_true_label]
                
            # Normalize the combined similarities to get the percentages
            percentages_with_true_label = combined_similarities / combined_similarities.sum()
                
            # Exclude the true label's percentage
            percentages = percentages_with_true_label[1:]
                
            if self.label_distribution:
                # Combine label with probability
                combined = [f" but maybe {prob*100:.2f} probability is {self.text_label[i]}" for i, prob in zip(top_similar_indices, percentages)]
                final_labels.append(combined)
            else:
                final_labels.append([self.text_label[i] for i in top_similar_indices])

        return final_labels
    
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

        # print(f"Using dataset: {type(self.dataset).__name__}")
        # if hasattr(self.dataset, 'class_id_to_name'):
        #     print("Dataset has class_id_to_name attribute.")
        # else:
        #     print("Dataset lacks class_id_to_name attribute.")

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