import numpy as np
import torch 
from torch import nn, Tensor
from tqdm import trange
from typing import Set, Union, List, Dict, Callable

from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import flatten2DList

class Evaluator(nn.Module):
    def __init__(self, model, tokenizer, word_segmenter) -> None:
        super(Evaluator, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.word_segmenter = word_segmenter
        
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Perform pooling with model outputs
        pooler_output = self.mean_pooling(outputs, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:,-1,:]
        
        return {
            "pooler_output": pooler_output,
            "last_hidden": last_hidden
        }

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               encoded_type: str="pooler_output") -> Union[List[Tensor], np.ndarray, Tensor]:
        '''
        Get multiple embedding of sentences
        '''
        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False

        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.to(device)
        
        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size, desc="Processing", disable=not show_progress_bar):
            sentence_batch = sentences[start_index: start_index+batch_size]
            segmented_batch = []
            for sent in sentence_batch:
                try:
                    segmented_sent = " ".join(flatten2DList(self.word_segmenter.tokenize(sent)))
                except:
                    segmented_sent = sent
                segmented_batch.append(segmented_sent)
 
            features = self.tokenizer(segmented_batch,
                       padding='max_length', 
                       truncation=True, 
                       max_length=300,
                       return_tensors='pt').to(device)
            
            with torch.no_grad():
                out_features = self.forward(input_ids=features["input_ids"], 
                                   attention_mask=features["attention_mask"])
                embeddings = []
                for row in out_features[encoded_type]:
                    embeddings.append(row.cpu())
                all_embeddings.extend(embeddings)

        if convert_to_tensor:
            all_embeddings = torch.vstack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__=="__main__":
    pass