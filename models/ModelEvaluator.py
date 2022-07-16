import numpy as np
import torch 
from torch import nn, Tensor
from tqdm import trange
from typing import Set, Union, List, Dict, Callable
from utils import flatten2DList, get_device, mean_pooling

class Evaluator(nn.Module):
    def __init__(self, model, tokenizer, word_segmenter) -> None:
        super(Evaluator, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.word_segmenter = word_segmenter
        
    def forward(self, inputs, from_huggingface=True):
        if from_huggingface:
            outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
            # Perform pooling with model outputs
            pooler_output = mean_pooling(outputs, attention_mask=inputs["attention_mask"])
            return pooler_output
        outputs = self.model(inputs)
        return outputs
        
        

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               max_seq_len: int = 300,
               show_progress_bar: bool = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device=None,
               from_huggingface: bool=True) -> Union[List[Tensor], np.ndarray, Tensor]:
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
            device = get_device()

        self.to(device)
        input_keys = ["input_ids", "attention_mask"]
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
                       max_length=max_seq_len,
                       return_tensors='pt')

            input_feats = {k: features[k].to(device) for k in input_keys}
            with torch.no_grad():
                out_features = self.forward(input_feats, from_huggingface)
                embeddings = []
                for row in out_features:
                    embeddings.append(row.cpu())
                all_embeddings.extend(embeddings)

        if convert_to_tensor:
            all_embeddings = torch.vstack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

if __name__=="__main__":
    pass