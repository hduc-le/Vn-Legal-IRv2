import os
from turtle import forward
import torch
import argparse
import warnings

from torch import nn 
from models.MemoryModel import MemoryModel

from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer
)
from utils import *
from losses.ContrastiveLoss import SupervisedContrastiveLoss
warnings.filterwarnings('ignore')

class DataForCL(torch.utils.data.Dataset):
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

class Model(nn.Module):
    def __init__(self, model_name_or_path) -> None:
        super(Model, self).__init__()
        self.bart_model = AutoModel.from_pretrained(model_name_or_path)
        for param in self.bart_model.parameters():
            param.requires_grad = False 

        self.memory_model = MemoryModel(self.bart_model.config.hidden_size, self.bart_model.config.hidden_size)

    def forward(self, query_inputs, pos_inputs):
        with torch.no_grad():
            query_out = self.bart_model(
                input_ids=query_inputs["input_ids"],
                attention_mask=query_inputs["attention_mask"]
            )
            pos_out = self.bart_model(
                input_ids=pos_inputs["input_ids"],
                attention_mask=pos_inputs["attention_mask"]
            )

            query_emb = mean_pooling(query_out, query_inputs["attention_mask"])
            pos_emb = mean_pooling(pos_out, pos_inputs["attention_mask"])

        output_dict = self.memory_model(query_emb)
        query_transformed = output_dict["query_transformed"]
        return query_transformed, pos_emb

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_data", default="./generated_data/train_pairs.pkl", type=str, help="path to sentence-pairs for contrastive training")
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-word", type=str)
    parser.add_argument("--mem_size", default=10, type=int)
    parser.add_argument("--saved_model", default="saved_model/model-memory", type=str)
    parser.add_argument("--max_seq_len", default=300, type=int)
    parser.add_argument("--temperature", default=0.1, type=float, help="hyper-parameter for contrastive loss")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_decay", default=False, type=bool)
    parser.add_argument("--decay_rate", default=0.96, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--save_model",  default='finish', const='finish', nargs='?', choices=['epoch', 'finish'], help="save model every epoch (`epoch`) or end of training (`finish`), (default: %(default)s)")
    args = parser.parse_args()
    
    device = get_device()

    print(">> Preparing paired data for contrastive learning.")
    segmented_pairs = load_parameter(args.paired_data)

    examples0 = [sent for sent, _ in segmented_pairs]
    examples1 = [sent for _, sent in segmented_pairs]

    print(">> Download pretrained tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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

    train_dataset = DataForCL(encodings0, encodings1)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    

    model = MemoryModel(model_name_or_path=args.model_name_or_path, memory_size=args.mem_size)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = SupervisedContrastiveLoss(args.temperature)
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=args.decay_rate)

    input_keys = ["input_ids", "attention_mask"]
    
    print("============= Start training =============")
    for epoch in range(args.num_epochs):
        overall_loss = 0.0
        iterator = tqdm(train_loader, leave=True)
        model.train()
        for query_features, sentence_features in iterator:
            # clear gradient
            optim.zero_grad()
            query_inputs = {key: query_features[key].to(device) for key in input_keys}
            pos_inputs = {key: sentence_features[key].to(device) for key in input_keys}
            output_dict = model(query_inputs, pos_inputs)
            
            # compute loss
            if args.regularization:
                loss = loss_fn(output_dict["query_transformed"], output_dict["positive_emb"]) + 0.01*torch.norm(output_dict["query_attn_weights"])
            else:
                loss = loss_fn(output_dict["query_transformed"], output_dict["positive_emb"])
            
            overall_loss += loss.item()

            # backpropagation
            loss.backward()
            optim.step()

            iterator.set_description('Epoch {}'.format(epoch))
            iterator.set_postfix(loss=loss.item())

        print(f'Finished epoch {epoch}.')
        print(">> Epoch loss: {:.5f}".format(overall_loss/len(train_loader)))

        if args.lr_decay:
            lr_scheduler.step()
        if args.save_model == 'epoch':
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict()
            }, os.path.join(args.saved_model, "epoch={} model-memory.pth".format(epoch)))
        
    if args.save_model == 'finish':
        torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict()
            }, os.path.join(args.saved_model, "model-memory.pth"))
