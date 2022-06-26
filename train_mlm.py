import os
import torch 
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.notebook import trange
from utils import *
from vncorenlp import VnCoreNLP

logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p %Z",
    level = logging.INFO
)
device = get_device()

class DataForMLM(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        features = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        return features
    def __len__(self):
        return len(self.encodings['input_ids'])

def prepare_mlm_input_and_labels(encodings, tokenizer, mlm_probability=0.15):
    encodings['labels'] = encodings['input_ids'].detach().clone()
    rand = torch.rand(encodings['input_ids'].shape)
    mask_arr = (rand < mlm_probability) \
                * (encodings['input_ids'] != tokenizer.convert_tokens_to_ids(tokenizer.cls_token)) \
                * (encodings['input_ids'] != tokenizer.convert_tokens_to_ids(tokenizer.eos_token)) \
                * (encodings['input_ids'] != tokenizer.convert_tokens_to_ids(tokenizer.pad_token)) 
    selection = []
    for i in trange((mask_arr.shape[0]), desc='Masking'):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(mask_arr.shape[0]):
        encodings['input_ids'][i, selection[i]] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return encodings


def train(model, optim, dataloader, epoch):
    iterator = tqdm(dataloader, leave=True)
    overall_loss = 0.0
    model.train()
    for batch in iterator:
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        
        loss = outputs.loss
        overall_loss += loss.item()
        
        loss.backward()
        optim.step()
        
        iterator.set_description('Epoch: {} - training'.format(epoch))
        iterator.set_postfix(loss=loss.item())
    overall_loss /= len(dataloader)
    return overall_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legal_data", default="./generated_data", type=str, help="path to segmented docs for mlm training")
    parser.add_argument("--saved_model", default="./saved_model/model-mlm", type=str)
    parser.add_argument("--model_name_or_path", default="vinai/bartpho-word", type=str)
    parser.add_argument("--max_seq_len", default=300, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_decay", default=False, type=bool)
    parser.add_argument("--decay_rate", default=0.96, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)

    args = parser.parse_args()
    
    segmented_docs = load_parameter(os.path.join(args.legal_data, "segmented_docs.pkl"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    encodings = tokenizer(
        segmented_docs,
        truncation=True,
        padding="max_length",
        max_length=args.max_seq_len,
        return_tensors="pt"
    )
    encodings = prepare_mlm_input_and_labels(encodings, tokenizer, mlm_probability=0.15)
    dataset = DataForMLM(encodings)
    dataloader = torch.utils.data.DataLoader(dataset,   
                                         batch_size=args.batch_size,
                                         shuffle=True)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=args.decay_rate)

    
    logging.info("Start training...")
    for epoch in range(args.num_epochs):
        loss = train(model, optim, dataloader, epoch) 
        logging(f'Finished epoch {epoch}.')
        logging("Epoch loss: {:.5f}".format(loss))
        if args.lr_decay:
            lr_scheduler.step()

        logging.info("Saving the model and tokenizer")
        model.save_pretrained(args.saved_model)
        tokenizer.save_pretrained(args.saved_model)
    
    logging.info("Done.")

if __name__=="__main__":
    main()
    
