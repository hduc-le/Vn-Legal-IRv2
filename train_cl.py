import os
import torch 
import logging
import argparse
import warnings

from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer
)
from utils import *
from vncorenlp import VnCoreNLP
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity()
    def _eval_denom(self, z1, z2):
        cosine_vals = []
        for v in z1:
            cosine_vals.append(self.sim(v.view(1,-1), z2)/self.temperature)
        cos_batch = torch.cat(cosine_vals, dim=0).view(z1.shape[0], -1)
        denom = torch.sum(torch.exp(cos_batch),dim=1)
        return denom
    def _contrastive_loss(self, z1, z2):
        num = torch.exp(self.sim(z1, z2)/self.temperature)
        denom = self._eval_denom(z1, z2)
        loss = -torch.mean(torch.log(num/denom))
        return loss
    def forward(self, z1, z2):
        return self._contrastive_loss(z1, z2)

class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings0, encodings1):
        self.encodings0 = encodings0
        self.encodings1 = encodings1
    def __getitem__(self, idx):
        features0 = {
            key: torch.tensor(val[idx]) for key, val in self.encodings0.items()
        }
        features1 = {
            key: torch.tensor(val[idx]) for key, val in self.encodings1.items()
        }
        return features0, features1
    def __len__(self):
        return len(self.encodings0['input_ids'])

def train(model, optimizer, dataloader, epoch):
    overall_loss = 0.0
    loss_fct = SupervisedContrastiveLoss()
    iterator = tqdm(dataloader, leave=True)
    model.train()
    
    for ft0, ft1 in iterator:
        optimizer.zero_grad()

        input_ids0 = ft0["input_ids"].to(device)
        attention_mask0 = ft0["attention_mask"].to(device)
        model_output0 = model(input_ids=input_ids0, attention_mask=attention_mask0)

        input_ids1 = ft1["input_ids"].to(device)
        attention_mask1 = ft1["attention_mask"].to(device)
        model_output1 = model(input_ids=input_ids1, attention_mask=attention_mask1)

        # compute loss
        sent0_repr, sent1_repr = model_output0.last_hidden_state[:,-1,:], model_output1.last_hidden_state[:,-1,:]
        loss = loss_fct(sent0_repr, sent1_repr)
        overall_loss += loss.item()

        # backprop
        loss.backward()
        optimizer.step()

        # display training info
        iterator.set_description('Epoch {}'.format(epoch))
        iterator.set_postfix(loss=loss.item())

    overall_loss/len(dataloader)
    return overall_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="./data", type=str, help="for loading question")
    parser.add_argument("--legal_data", default="./generated_data", type=str, help="path to legal legal_dict for reference")
    parser.add_argument("--saved_model", default="./saved_model/model4cl-finetune-from-bartphomlm-original", type=str)
    parser.add_argument("--checkpoints", default="./checkpoints/", type=str)
    parser.add_argument("--word_segmentator", default="./VnCoreNLP/VnCoreNLP-1.1.1.jar", type=str)
    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-word", type=str)
    parser.add_argument("--max_seq_len", default=300, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_decay", default=False, type=bool)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    args = parser.parse_args()
    
    device = get_device()

    legal_data = load_json(os.path.join(args.raw_data, "legal_corpus.json"))
    train_question_answer = load_json(os.path.join(args.raw_data, "train_question_answer.json"))
    legal_dict  = load_json(os.path.join(args.legal_data, "legal_dict.json"))
    
    qa_train, qa_test = train_test_split(train_question_answer["items"], test_size=1-args.train_ratio, random_state=42)

    os.makedirs(args.checkpoints, exist_ok=True)
    print("Saving test set at {}.".format(args.checkpoints))
    save_parameter(qa_test, os.path.join(args.checkpoints, "qa_test.pkl"))

    print("Pairing the contrastive setences.")
    qa_pairs = [
        [clean_text(item["question"]), legal_dict[label["law_id"]+"@"+label["article_id"]]]\
            for item in tqdm(qa_train, desc="Pairing") for label in item["relevant_articles"]
    ]
    
    print("Perform word-segmentation.")
    print("Load annotator.")
    annotator = VnCoreNLP(
        args.word_segmentator, 
        annotators="wseg,pos,ner,parse", 
        max_heap_size="-Xmx2g"
    )
    segmented_pairs = []
    for pair in tqdm(qa_pairs, desc="processing"):
        try:
            sent0 = " ".join(flatten2DList(annotator.tokenize(pair[0])))
            sent1 = " ".join(flatten2DList(annotator.tokenize(pair[1])))
        except:
            sent0 = pair[0]
            sent1 = pair[1]
        segmented_pairs.append(
            [sent0, sent1]
        )

    examples0 = [sent for sent, _ in segmented_pairs]
    examples1 = [sent for _, sent in segmented_pairs]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)

    encodings0 = tokenizer(
        examples0,
        truncation=True,
        padding="max_length",
        max_length=args.max_seq_len,
        return_tensors="pt"
    )
    encodings1 = tokenizer(
        examples1,
        truncation=True,
        padding="max_length",
        max_length=args.max_seq_len,
        return_tensors="pt"
    )

    train_dataset = LegalDataset(encodings0, encodings1)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.96)
    for epoch in range(args.num_epochs):
        loss = train(model, optim, train_loader, epoch)
        print(f'Finished epoch {epoch}.')
        print(">> Epoch loss: {:.5f}".format(loss))
        if args.lr_decay:
            lr_scheduler.step()
    model.save_pretrained(os.path.join(args.saved_model))
