import os
import torch
import argparse
import warnings
import torch.nn.functional as F
from torch import nn 
from models.MemoryModel import MemoryModel

from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer
)
from vncorenlp import VnCoreNLP
from utils import *
warnings.filterwarnings('ignore')


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_pairs, tokenizer):
        self.train_pairs = train_pairs
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        sent = self.train_pairs[idx][0]
        sent_label = self.train_pairs[idx][1]
        retrieved_top = self.train_pairs[idx][2]
        groundtruths = self.train_pairs[idx][3]

        sent_feats = self.tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
            return_tensors="pt"
        )
        sent_labels_feats = self.tokenizer(
            sent_label,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
            return_tensors="pt"
        )
        retr_feats = self.tokenizer(
            retrieved_top,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
            return_tensors="pt"
        )
        gt_feats = self.tokenizer(
            groundtruths,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
            return_tensors="pt"
        )
        return {"feats0": sent_feats, "feats1": sent_labels_feats,"feats2": retr_feats, "feats3": gt_feats}
    def __len__(self):
        return len(self.train_pairs)

def make_pair_question_answer(QA_set, annotator):
    pairs = []
    for item in tqdm(QA_set):
        try:
            seg_question = " ".join(flatten2DList(annotator.tokenize(clean_text(item["question"]))))
            for label in item["relevant_articles"]:
                pos_sent = legal_dict[label["law_id"]+"@"+label["article_id"]]
                seg_pos_sent = " ".join(flatten2DList(annotator.tokenize(pos_sent)))
                pairs.append(
                    [seg_question, seg_pos_sent]
                )
        except:
            pass
    return pairs

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./raw_data/", type=str, help="path to question answer set")
    parser.add_argument("--generated_data", default="vinai/bartpho-word", type=str)
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-word", type=str)
    parser.add_argument("--saved_model", default="saved_model/model-memory.pth", type=str)
    parser.add_argument("--max_seq_len", default=300, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    args = parser.parse_args()
    
    device = get_device()

    annotator = VnCoreNLP(
        "./VnCoreNLP/VnCoreNLP-1.1.1.jar", 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )

    legal_dict = load_json(os.path.join(args.generated_data, "legal_dict.json"))
    Q_memories = load_parameter(os.path.join(args.generated_data, "Q_memories.pkl"))
    Q_new = load_parameter(os.path.join(args.generated_dat, "Q_new.pkl"))
    
    bm25_query_refers = np.array([sent for sent, _ in Q_memories])
    bm25_positive_refers = np.array([pos_sent for _, pos_sent in Q_memories])

    bm25 = load_parameter(os.path.join(args.saved_model, "bm25_model.pkl"))

    train_pairs = []
    for q_new, q_new_label in tqdm(Q_new):
        scores = bm25.get_scores(q_new.split())
        top10_ids = np.argsort(scores)[::-1][:10]
        retrieved_queries = bm25_query_refers[top10_ids].tolist()
        positive_list = bm25_positive_refers[top10_ids].tolist()
        train_pairs.append(
            [q_new, q_new_label, retrieved_queries, positive_list]
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = Dataset(train_pairs, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=1,
                                        shuffle=True)

    model = MemoryModel()
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    input_keys = ["input_ids", "attention_mask"]
    for epoch in range(1):
        overall_loss = 0.0
        iterator = tqdm(train_loader, leave=True)
        model.train()
        
        for batch in iterator:
            # clear gradient
            optim.zero_grad()

            query_features = batch["feats0"]
            query_features = {key: query_features[key].view(-1, query_features[key].shape[-1]).to(device) for key in input_keys}

            query_label_features = batch["feats1"]
            query_label_features = {key: query_label_features[key].view(-1, query_label_features[key].shape[-1]).to(device) for key in input_keys}

            bm25_retr_features = batch["feats2"]
            bm25_retr_features = {key: bm25_retr_features[key].view(-1, bm25_retr_features[key].shape[-1]).to(device) for key in input_keys}

            bm25_gt_features = batch["feats3"]
            bm25_retr_features = {key: bm25_gt_features[key].view(-1, bm25_gt_features[key].shape[-1]).to(device) for key in input_keys}
            
            query_transformed, query_label_emb = model(query_features, query_label_features, bm25_retr_features, bm25_gt_features)
            
            # compute loss
            loss = 1-cos(query_transformed, query_label_emb)
            overall_loss += loss.item()

            # backpropagation
            loss.backward()
            optim.step()

            iterator.set_description('Epoch {}'.format(epoch))
            iterator.set_postfix(loss=loss.item())
            
        logging.info(f'Finished epoch {epoch}.')
        logging.info("Epoch loss: {:.5f}".format(overall_loss/len(train_loader)))
    
    