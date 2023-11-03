import open_clip
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import custom_collate_fn, rough_descriptions
import random



class Enhancement:
    def __init__(
        self,
        dataset,
        text_label,
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
        label_distribution=False,
        rough_desc=False,
        only_probability=False,
    ):
        self.dataset = dataset
        self.text_label = text_label
        self.device = device
        self.batch_size = batch_size
        self.label_distribution = label_distribution
        self.rough_desc = rough_desc
        self.only_probability = only_probability
        
        # Load the model and tokenizer
        self.model, _, self.image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(vision_encoder_path)

        # Tokenize the text descriptions and move to device
        self.text_tokens = self.tokenizer(text_label).to(self.device)
        self.text_features = cached_features
    
    def Multilabel(self, batch, true_labels, num_examples):
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
            final_indices_with_true_label = []
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
                final_indices_with_true_label.append(similar_indices_with_true_label)

        if self.label_distribution:
            combined_texts = []
            for label_row, prob_row in zip(final_indices, final_probs):
                combined = [f" but may have { prob*100:.2f}% probability of being {self.text_label[i]}" for i, prob in zip(label_row, prob_row)]
                combined_texts.append(combined)
            return combined_texts
        elif self.rough_desc:
            # Select prompt template 
            prompt_text_template = rough_descriptions[num_examples]

            combined_texts = []
            for _, label_row in enumerate(final_indices_with_true_label):
                text_labels = [self.text_label[i] for i in label_row]
                # Fill in the blanks 
                combined = prompt_text_template.format(*text_labels)
                combined_texts.append(combined)
            return combined_texts
        else:
            similar_texts = [[self.text_label[i] for i in row] for row in final_indices]
            return similar_texts
        
    def find_similar_labels(self, query_texts, num_similar=1):
        """
        Given specific labels, find the similiar ones.
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
                combined = [f" but may have {prob*100:.2f}% probability of being {self.text_label[i]}" for i, prob in zip(top_similar_indices, percentages)]
                final_labels.append(combined)
            else:
                final_labels.append([self.text_label[i] for i in top_similar_indices])

        return final_labels
    
    def MultilabelwithSimilarity(self, batch, true_labels, num_examples):
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

            # Map into [0,1]  
            similarities = (similarities + 1) / 2
            
            # Get the indices of the 'num_examples' most similar texts
            sorted_indices = similarities.argsort(dim=-1, descending=True)
            
            final_indices = []
            final_indices_with_true_label = []
            final_probs = []
            for idx, true_label in enumerate(true_labels):
                # Get indices of the top 'num_examples' similar labels excluding the true label
                top_similar_indices = [i for i in sorted_indices[idx] if self.text_label[i] != true_label][:num_examples]

                # Extract the original similarity values for these indices
                top_similarities = similarities[idx, top_similar_indices]
                
                final_probs.append(top_similarities)
                final_indices.append(top_similar_indices)

        if self.label_distribution:
            combined_texts = []
            for label_row, prob_row in zip(final_indices, final_probs):
                if self.only_probability:
                    combined = [f" { prob*100:.2f}%{self.text_label[i]}" for i, prob in zip(label_row, prob_row)]
                else:
                    combined = [f" but may have { prob*100:.2f}% probability of being {self.text_label[i]}" for i, prob in zip(label_row, prob_row)]
                combined_texts.append(combined)
            return combined_texts
        elif self.rough_desc:
            # Select prompt template 
            prompt_text_template = rough_descriptions[num_examples]

            combined_texts = []
            for _, label_row in enumerate(final_indices_with_true_label):
                text_labels = [self.text_label[i] for i in label_row]
                # Fill in the blanks 
                combined = prompt_text_template.format(*text_labels)
                combined_texts.append(combined)
            return combined_texts
        else:
            similar_texts = [[self.text_label[i] for i in row] for row in final_indices]
            return similar_texts
        
    def find_similar_labelswithSimilarity(self, query_texts, num_similar=1):
        """
        Given specific labels, find the similiar ones.
        """
        # Change into Evaluation Mode
        self.model.eval()

        with torch.no_grad():
            # tokenizaiton
            query_tokens = self.tokenizer(query_texts).to(self.device)
            query_features = self.model.encode_text(query_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_features = query_features.detach().cpu()

            # Compute similarity among labels and query true labels
            similarities = (query_features @ self.text_features.T).squeeze()
            if len(similarities.shape) == 1:  # Single Query
                similarities = similarities.unsqueeze(0)
            
            # map into [0,1]
            similarities = (similarities + 1) / 2

            # Get the similar ones
            sorted_indices = similarities.argsort(dim=-1, descending=True)
            
        # Make sure is diffrenet with GT
        final_labels = []
        for idx, query_text in enumerate(query_texts):
            # Get indices of the top 'num_similar' similar labels excluding the true label
            top_similar_indices = [i for i in sorted_indices[idx] if self.text_label[i] != query_text][:num_similar]

            # Exclude the true label's percentage
            percentages = similarities[idx, top_similar_indices]
                
            if self.label_distribution:
                # Combine label with probability
                if self.only_probability:
                    combined = [f" {prob*100:.2f}%{self.text_label[i]}" for i, prob in zip(top_similar_indices, percentages)]
                else:
                    combined = [f" but may have {prob*100:.2f}% probability of being {self.text_label[i]}" for i, prob in zip(top_similar_indices, percentages)]
                final_labels.append(combined)
            elif self.rough_desc:
                combined = [f" but may be a {self.text_label[idx]}" for _, idx in enumerate(top_similar_indices)]
                final_labels.append(combined)
            else:
                final_labels.append([self.text_label[i] for i in top_similar_indices])

        return final_labels
    
    
    # def Multilabel2labels(self, batch, true_labels, num_examples):
    #     """
    #     Get the ranked text descriptions based on similarity to the image but 
    #     avoid returning labels that match the true_labels of the images.
    #     """
    #     self.model.eval()

    #     with torch.no_grad():
    #         # Preprocess and encode the image
    #         image_input = torch.stack([self.image_processor(image) for image in batch]).to(self.device)
            
    #         # Get the feature of the input image
    #         query_feature = self.model.encode_image(image_input)
    #         query_feature /= query_feature.norm(dim=-1, keepdim=True)
    #         query_feature = query_feature.detach().cpu()

    #         # Compute the similarity between image and text representations
    #         similarities = (query_feature @ self.text_features.T)

    #         for idx, true_label in enumerate(true_labels):
    #             true_label_index = self.text_label.index(true_label)
    #             similarities[idx, true_label_index] = -1e10
    #         amplified_similarities = similarities * 50
    #         # Compute probabilities using softmax
    #         probabilities = F.softmax(amplified_similarities, dim=-1)

    #         # Sort the probabilities if needed
    #         sorted_indices = probabilities.argsort(dim=-1, descending=True)

    #         final_indices = []
    #         final_indices_with_true_label = []
    #         final_probs = []
    #         for idx, true_label in enumerate(true_labels):
    #             # Get indices of the top 'num_examples' similar labels excluding the true label
    #             top_similar_indices = [i for i in sorted_indices[idx] if self.text_label[i] != true_label][:num_examples]
                
    #             # # Include the index of the true label
    #             # true_label_index = self.text_label.index(true_label)
    #             # similar_indices_with_true_label = [true_label_index] + top_similar_indices

    #             # Extract the original similarity values for these indices
    #             combined_similarities = probabilities[idx, top_similar_indices]
                
    #             # # Normalize the combined similarities to get the percentages
    #             # percentages_with_true_label = combined_similarities / combined_similarities.sum()
                
    #             # Exclude the true label's percentage
    #             percentages = combined_similarities
    #             final_probs.append(percentages)
    #             final_indices.append(top_similar_indices)
    #             # final_indices_with_true_label.append(similar_indices_with_true_label)

    #     if self.label_distribution:
    #         combined_texts = []
    #         for label_row, prob_row in zip(final_indices, final_probs):
    #             combined = [f" but may have { prob*100:.2f}% probability of being {self.text_label[i]}" for i, prob in zip(label_row, prob_row)]
    #             combined_texts.append(combined)
    #         return combined_texts
    #     elif self.rough_desc:
    #         # Select prompt template 
    #         prompt_text_template = rough_descriptions[num_examples]

    #         combined_texts = []
    #         for _, label_row in enumerate(final_indices_with_true_label):
    #             text_labels = [self.text_label[i] for i in label_row]
    #             # Fill in the blanks 
    #             combined = prompt_text_template.format(*text_labels)
    #             combined_texts.append(combined)
    #         return combined_texts
    #     else:
    #         similar_texts = [[self.text_label[i] for i in row] for row in final_indices]
    #         return similar_texts
        
    # def find_similar_labels2(self, query_texts, num_similar=1):
    #     """
    #     Given specific labels, find the similiar ones.
    #     """
    #     # 切换到评估模式
    #     self.model.eval()

    #     with torch.no_grad():
    #         # 对查询文本进行分词并移至设备
    #         query_tokens = self.tokenizer(query_texts).to(self.device)
    #         query_features = self.model.encode_text(query_tokens)
    #         query_features /= query_features.norm(dim=-1, keepdim=True)
    #         query_features = query_features.detach().cpu()

    #         # 计算查询文本和所有标签之间的相似性
    #         similarities = (query_features @ self.text_features.T).squeeze()
    #         if len(similarities.shape) == 1:  # 处理单个查询文本的情况
    #             similarities = similarities.unsqueeze(0)
                
    #         for idx, true_label in enumerate(query_texts):
    #             true_label_index = self.text_label.index(true_label)
    #             similarities[idx, true_label_index] = -1e10
                
    #         # Amplify the differences between the scores
    #         amplified_similarities = similarities * 50
            
    #         # Compute probabilities using softmax
    #         probabilities = F.softmax(amplified_similarities, dim=-1)
            
    #         # Get the indices of the 'num_examples' most similar (or probable) texts
    #         sorted_indices = probabilities.argsort(dim=-1, descending=True)
            
    #     final_labels = []
    #     for idx, query_text in enumerate(query_texts):
    #         # Get indices of the top 'num_similar' similar labels excluding the true label
    #         top_similar_indices = [i for i in sorted_indices[idx] if self.text_label[i] != query_text][:num_similar]

    #         # # Include the index of the true label
    #         # true_label_index = self.text_label.index(query_text)
    #         # similar_indices_with_true_label = [true_label_index] + top_similar_indices

    #         # Extract the original similarity values for these indices
    #         combined_similarities = probabilities[idx, top_similar_indices]
                
    #         # # Normalize the combined similarities to get the percentagescle
    #         # percentages_with_true_label = combined_similarities / combined_similarities.sum()
                
    #         # Exclude the true label's percentage
    #         percentages = combined_similarities
                
    #         if self.label_distribution:
    #             # Combine label with probability
    #             combined = [f" but may have {prob*100:.2f}% probability of being {self.text_label[i]}" for i, prob in zip(top_similar_indices, percentages)]
    #             final_labels.append(combined)
    #         else:
    #             final_labels.append([self.text_label[i] for i in top_similar_indices])

    #     return final_labels
    