import os
import torch 
import argparse
import warnings
import logging

from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer
)
from utils import *
from losses.ContrastiveLoss import SupervisedContrastiveLoss
warnings.filterwarnings('ignore')

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)

class DataForCL(torch.utils.data.Dataset):
    def __init__(self, encodings0, encodings1, encodings2=None):
        self.encodings0 = encodings0
        self.encodings1 = encodings1
        self.encodings2 = encodings2
    def __getitem__(self, idx):
        features0 = {
            key: torch.tensor(val[idx]) for key, val in self.encodings0.items()
        }
        features1 = {
            key: torch.tensor(val[idx]) for key, val in self.encodings1.items()
        }
        if self.encodings2 is not None:
            features2 = {
                key: torch.tensor(val[idx]) for key, val in self.encodings2.items()
            }
            return {"feats0":features0, "feats1": features1, "feats2":features2}
        return {"feats0": features0, "feats1": features1}
    def __len__(self):
        return len(self.encodings0['input_ids'])

def train(model, optimizer, dataloader, epoch):
    overall_loss = 0.0
    loss_fct = SupervisedContrastiveLoss(args.temperature)
    iterator = tqdm(dataloader, leave=True)
    sent2_repr = None
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()

        if len(batch) == 2:
            ft0, ft1 = batch["feats0"], batch["feats1"]
        elif len(batch) == 3:
            ft0, ft1, ft2 = batch["feats0"], batch["feats1"], batch["feats2"]
            input_ids2 = ft2["input_ids"].to(device)
            attention_mask2 = ft2["attention_mask"].to(device)
            model_output2 = model(input_ids=input_ids2, attention_mask=attention_mask2)
            sent2_repr = mean_pooling(model_output2, attention_mask2)

        input_ids0 = ft0["input_ids"].to(device)
        attention_mask0 = ft0["attention_mask"].to(device)
        model_output0 = model(input_ids=input_ids0, attention_mask=attention_mask0)

        input_ids1 = ft1["input_ids"].to(device)
        attention_mask1 = ft1["attention_mask"].to(device)
        model_output1 = model(input_ids=input_ids1, attention_mask=attention_mask1)

        # compute loss
        
        sent0_repr, sent1_repr = mean_pooling(model_output0, attention_mask0), mean_pooling(model_output1, attention_mask1)
        
        loss = loss_fct(sent0_repr, sent1_repr, sent2_repr)
        overall_loss += loss.item()

        # backprop
        loss.backward()
        optimizer.step()

        # display training info
        iterator.set_description('Epoch {}'.format(epoch))
        iterator.set_postfix(loss=loss.item())

    overall_loss = overall_loss/len(dataloader)
    return overall_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_data", default="./generated_data", type=str, help="path to sentence-pairs for contrastive training")
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-word", type=str)
    parser.add_argument("--saved_model", default="./saved_model", type=str)
    parser.add_argument("--max_seq_len", default=300, type=int)
    parser.add_argument("--temperature", default=0.1, type=float, help="hyper-parameter for contrastive loss")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_decay", default=False, type=bool)
    parser.add_argument("--decay_rate", default=0.96, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    
    args = parser.parse_args()
    
    device = get_device()

    logging.info("Preparing paired data for contrastive learning.")
    segmented_pairs = load_parameter(os.path.join(args.generated_data, "train_cl_pairs.pkl"))
    num_sent = len(segmented_pairs[0])

    logging.info("Download pretrained tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    encodings2 = None
    if num_sent == 2:
        examples0 = [sent for sent, _ in segmented_pairs]
        examples1 = [sent for _, sent in segmented_pairs]

    elif num_sent == 3:
        examples0 = [sent for sent, _, _ in segmented_pairs]
        examples1 = [sent for _, sent, _ in segmented_pairs]
        examples2 = [sent for _, _, sent in segmented_pairs]

        encodings2 = tokenizer(
            examples2,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
            return_tensors="pt"
        )
    
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
        
    train_dataset = DataForCL(encodings0, encodings1, encodings2)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

    logging.info("Download pretrained model")
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=args.decay_rate)

    logging.info("Start Training")
    for epoch in range(args.num_epochs):
        loss = train(model, optim, train_loader, epoch)
        logging.info(f'Finished epoch {epoch}.')
        logging.info("Epoch loss: {:.5f}".format(loss))
        if args.lr_decay:
            lr_scheduler.step()

    os.makedirs("./saved_model", exist_ok=True)       
    model.save_pretrained(os.path.join(args.saved_model, "model-cl"))
    tokenizer.save_pretrained(os.path.join(args.saved_model, "model-cl"))
