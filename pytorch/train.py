import tez 
import torch
import transformers 
import numpy as np 
import torch.nn as nn
import pandas as pd 
from sklearn import metrics 
from transformers import AdamW, get_linear_schedule_with_warmup

class BERTDataset:
    def__init__(self, texts, targets, max_len=64):
        self.texts = texts 
        self.targets = targets 
        self.max_len = max_len 
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncases",
            do_lower_case=False
        )

    def __len__(self):
        return len(self.texts)

    def __getitem___(self, idx):
        """return item on index passed \
        in encode_plus() we can put a many input \
        texts as required. For just a single sentence 
        we'll add None as a second argument"""

        text =  str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True, # adds [CLS] and [SEP] tokens
            max_len=self.max_len,
            padding="PAD",
            truncation=True
        )

        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "ids": torch.tensor(self.targets[idx], dtype=torch.float)
        }

        return resp 


# build the model 
class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased",
            return_dict=False
        )

        self.bert_drop = nn.Drop(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(),
                    lr=1e-4)

        return opt 

    def fetch_scheduler(self):
        sch = get_linear_schedular_with_warmup(
            self.optimizer, 
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps
        )

        return sch 

    def loss(self, outputs ,targets):
        return nn.BCEWithLogisLoss()(outpus, targets.view(-1,1))

    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detch().numpy >= 0.5
        targets = targets.cpu().detch().numpy

        return {
            "accuracy": metrics.accuracy_score(
                targets, outputs
            )
        }

    def forward(self, ids, maks, token_type_ids, targets=None):
        """it takes arguments same as in out dataset
        e.g, ids, mask, token_type_ids, targets"""
        
        # pass input to the BERT model
        _, x = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,

        )
        x = self.bert_drop(x)
        x = self.out(x)
        
        # calculate the loss. This can be done only when you
        # have targets available 

        if targets is not None:
            loss = self.loss(ouputs ,targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met 

        else:
            return x, 0, {}

def train_model(fold):
    df = pd.read_csv("")
    df_train = df_train[df != fold].reset_index(drop=True)
    # df_train = df_train[df_train != fold].reset_index(drop=True)
    df_valid = df_valid[df.kfold == fold].reset_index(drop=True)

    train_dataset = BERTDataset(
        df_train.review.values,
        df_train.sentiment.values
    )

    valid_dataset = BERTDataset(
        df_valid.review.values,
        df_valid.sentiment.values
    )


    n_train_steps = int(len(df_train )/ TRAIN_BS * EPOCHS)
    model = TextModel(
        num_classes=1,
        num_train_steps=n_train_steps,
        
    )

    es = tez.callbacks.EarlyStopping(monitor="valid_loss", patience=3)
    model.fit(train_dataset, valid_dataset=valid_dataset,
    device="cpu", epochs=10, train_bs=32,
    callbacks=[es])


if __name__ == '__main__':
    train_model(fold=0)

