import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import  TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
try:
    from keras.preprocessing.sequence import pad_sequences
except:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
'''
    print(f'{e}\nTry:\npip install tensorflow==2.0.0')
'''
from tqdm import tqdm, trange
tqdm.pandas()


class SWOT_ANALYSIS():
    def __init__(
        self,
        model_type,input_json_file,batch_size=16,device_type=None,
        epochs=4,labels_count=5,MaxLen=480,
        pad_type='post',trunc_type='post',
        test_size=0.2,valid_size=0.2,random_state=27):
        self.model_type = model_type # 'distilbert', 'bert', 'electra', 'roberta'
        self.model_path = f'model_{model_type}'
        self.swot_data_file = input_json_file #'swot-training-data_min_15-max_355-splitter-0-36_date-2021-06-29.json'
        self.device = self.get_device(device_type)
        self.batch_size = batch_size
        self.epochs = epochs #2,3,4
        self.labels_count = labels_count
        self.MaxLen = MaxLen
        self.pad_type = pad_type
        self.trunc_type = trunc_type
        self.test_size = test_size
        self.valid_size = valid_size
        self.random_state = random_state
        if self.model_type in ['bert', 'roberta', 'electra']:
            self.batch_size = 8
        

    def get_device(self, device_type=None):
        if device_type:
            device = torch.device(device_type)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == 'cuda':
                n_gpu = torch.cuda.device_count()
                print('GPU NAME :', torch.cuda.get_device_name(0))
        return device

    def get_category_lables(self):
        lable_mapping = {
            "opportunities" :0,
            "threats": 1,
            "weaknesses": 2,
            "strengths": 3,
            "irrelevent": 4
        }
        return lable_mapping

    def read_training_data(self):
        with open(self.swot_data_file, 'r') as f:
            swot_data = json.loads(json.load(f))
        swot_train_df = pd.DataFrame(swot_data)
        swot_train_df = swot_train_df[(swot_train_df['category'] != 'relevent')]
        print(f'Category Value Count of Data : \n {swot_train_df["category"].value_counts()}')
        return swot_train_df

    def pretrained_tokenizer_and_model(self):
        print(f'Model Class : {self.model_type}')
        if self.model_type == 'bert':
            pretrained_model = 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_model, num_labels=self.labels_count)
        elif self.model_type == 'roberta':
            pretrained_model = 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
            self.model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model, num_labels=self.labels_count)
        elif self.model_type == 'distilbert':
            pretrained_model = 'distilbert-base-uncased'
            self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                pretrained_model, num_labels=self.labels_count)
        elif self.model_type == 'electra':
            pretrained_model = 'google/electra-small-discriminator'
            self.tokenizer = ElectraTokenizer.from_pretrained(pretrained_model)
            self.model = ElectraForSequenceClassification.from_pretrained(
                pretrained_model, num_labels=self.labels_count)
        if self.device.type == 'cuda': self.model.cuda()

    def clear_memory(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocessing_on_data(self, swot_df, train = False):
        if train:
            category_lable = self.get_category_lables()
            swot_df['label'] = swot_df['category'].apply(lambda c: category_lable[c])
            swot_df = swot_df[['description','category', 'label']]
            print(f"Mapping :\n{swot_df['category'].unique().tolist()}, {swot_df['label'].unique().tolist()}")
            print(f"Label value count:\n{swot_df['label'].value_counts()}")

        swot_df['tokensize_sent'] = swot_df['description'].apply(
            lambda s: self.tokenizer.tokenize(f'[CLS] {s} [SEP]'))

        swot_df['input_ids'] = swot_df['tokensize_sent'].apply(
            lambda t: pad_sequences(
                [self.tokenizer.convert_tokens_to_ids(t)], maxlen=self.MaxLen,
                dtype='int32', padding=self.pad_type, truncating=self.trunc_type)[0])

        swot_df['attention_mask'] = swot_df['input_ids'].apply(
            lambda ids: [float(id>0) for id in ids])
        return swot_df

    def get_train_test_data(self, swot_df):
        train_df, test_df = train_test_split(
            swot_df, test_size=self.test_size, 
            random_state=self.random_state, stratify=swot_df['label'])
        print(f"Split - \n Train {train_df['label'].value_counts().to_dict()}, \n Test {test_df['label'].value_counts().to_dict()}")
        return train_df, test_df

    def get_train_valid_data(self, train_df):
        train_df, valid_df = train_test_split(
            train_df, test_size=self.valid_size, 
            random_state=self.random_state, stratify=train_df['label'])
        print(f"Split - \n Train {train_df['label'].value_counts().to_dict()}, \n Valid {valid_df['label'].value_counts().to_dict()}")
        return train_df, valid_df

    def generate_dataloader(self, train_df, test_df=pd.DataFrame(), valid_df=pd.DataFrame()):
        train_dataset = TensorDataset(
            torch.tensor(train_df['input_ids'].to_list()),
            torch.tensor(train_df['attention_mask'].to_list()),
            torch.tensor(train_df['label'].to_list()))
        if not test_df.empty:
            test_dataset = TensorDataset(
                torch.tensor(test_df['input_ids'].to_list()),
                torch.tensor(test_df['attention_mask'].to_list()),
                torch.tensor(test_df['label'].to_list()))
        if not valid_df.empty:
            valid_dataset = TensorDataset(
                torch.tensor(valid_df['input_ids'].to_list()),
                torch.tensor(valid_df['attention_mask'].to_list()),
                torch.tensor(valid_df['label'].to_list()))

        train_sampler = RandomSampler(train_dataset)
        if not test_df.empty:
            test_sampler = SequentialSampler(test_dataset)
        if not valid_df.empty:
            valid_sampler = SequentialSampler(valid_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler)
        if not test_df.empty:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                sampler=test_sampler)
        if not valid_df.empty:
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                sampler=valid_sampler)
        if not test_df.empty and not valid_df.empty:
            return train_dataloader, test_dataloader, valid_dataloader
        elif not test_df.empty:
            return train_dataloader, test_dataloader
        elif not valid_df.empty:
            return train_dataloader, valid_dataloader
        return train_dataloader

    def generate_crawler_dataloader(self, crawler_df):
        crawler_dataset = TensorDataset(
            torch.tensor(crawler_df['input_ids'].to_list()),
            torch.tensor(crawler_df['attention_mask'].to_list()),
            torch.tensor(crawler_df['sentence_id'].to_list()))
        crawler_sampler = SequentialSampler(crawler_dataset)
        crawler_dataloader = DataLoader(
            crawler_dataset,
            batch_size=self.batch_size,
            sampler=crawler_sampler)
        return crawler_dataloader

    def get_optimizer_and_scheduler(self, train_dataloader):
        param_optimizer = list(self.model.named_parameters())
        if self.model_type == 'bert':
            no_decay = ['bias', 'LayerNorm.weight']
        elif self.model_type == 'distilbert':
            no_decay = ['bias', 'LayerNorm.weight']
        elif self.model_type == 'electra':
            no_decay = ['bias', 'LayerNorm.weight']
        elif self.model_type == 'roberta':
            no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.1},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,lr = 2e-5, eps = 1e-8)

        total_steps = len(train_dataloader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        return  optimizer, scheduler

    def accuracy_measure(self, true_labels,predictions):
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        return matthews_corrcoef(flat_true_labels, flat_predictions)

    def flat_accuracy(self, labels,preds):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, optimizer, scheduler, train_dataloader):
        for epoch in trange(self.epochs, desc='epochs'):
            #TRAINING
            epoch_tr_loss = 0
            nb_tr_steps = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(b.to(self.device) for b in batch)
                b_input_ids, b_input_masks, b_labels = batch
                #IMPORTANT
                optimizer.zero_grad()
                if self.model_type in ['bert', 'roberta','electra']:
                    output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks, labels=b_labels)
                elif self.model_type == 'distilbert':
                    output = self.model(b_input_ids, attention_mask=b_input_masks, labels=b_labels)
                loss = output['loss']
                #IMPORTANT
                loss.backward()
                #IMPORTANT
                optimizer.step()
                #IMPORTANT
                scheduler.step()
                epoch_tr_loss += loss.item()
                nb_tr_steps +=1
            print(f'training loss : {epoch_tr_loss/nb_tr_steps}')
            #EVALUATE
            epoch_eva_accuracy = 0
            nb_eva_steps = 0
            self.model.eval()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(b.to(self.device) for b in batch)
                b_input_ids, b_input_masks, b_labels = batch
                #IMPORANT
                with torch.no_grad():
                    if self.model_type in ['bert', 'roberta','electra']:
                        output = self.model(b_input_ids, attention_mask=b_input_masks, token_type_ids=None)
                    elif self.model_type == 'distilbert':
                        output = self.model(b_input_ids, attention_mask=b_input_masks)
                logits = output['logits'].detach().to('cpu').numpy()
                b_labels_cpu = b_labels.to('cpu').numpy()
                epoch_eva_accuracy += self.flat_accuracy(b_labels_cpu, logits)
                nb_eva_steps += 1
            print(f'eva accuracy : {epoch_eva_accuracy/nb_eva_steps}')
            self.save_swot_tokenizer_and_model(epoch)
        print('Training is Done')

    def evaluate(self, test_dataloader):
        self.model.eval()
        predictions , true_labels = [], []
        for step, batch in tqdm(enumerate(test_dataloader),
                                desc='Validating test results....',
                                total = len(test_dataloader)):
            batch = tuple(b.to(self.device) for b in batch)
            b_input_ids, b_input_masks, b_labels = batch
            #IMPORTANT
            with torch.no_grad():
                if self.model_type in ['bert', 'roberta','electra']:
                    output = self.model(b_input_ids, attention_mask=b_input_masks, token_type_ids=None)
                elif self.model_type == 'distilbert':
                    output = self.model(b_input_ids, attention_mask=b_input_masks)
            logits = output['logits'].detach().to('cpu').numpy()
            b_labels_cpu = b_labels.to('cpu').numpy()
            predictions.append(logits)
            true_labels.append(b_labels_cpu)
        accuracy = self.accuracy_measure(true_labels, predictions)
        print(f'accuracy on test dataset: {accuracy}')

    def finetune(self):
        try:
            swot_df = self.read_training_data()
            self.pretrained_tokenizer_and_model()
            swot_df = self.preprocessing_on_data(swot_df, train=True)
            train_df, test_df  = self.get_train_test_data(swot_df)
            train_dataloader, test_dataloader = self.generate_dataloader(train_df, test_df)
            optimizer, scheduler = self.get_optimizer_and_scheduler(train_dataloader)
            self.train(optimizer, scheduler,train_dataloader)
            self.evaluate(test_dataloader)
        finally:
            print(f'Check folder -- {self.model_path}')

    def save_swot_tokenizer_and_model(self, epoch):
        if not os.path.exists(f'{self.model_path}'):
            os.mkdir(self.model_path)
        if not os.path.exists(f'{self.model_path}/{epoch}'):
            os.mkdir(f'{self.model_path}/{epoch}')
        torch.save(self.tokenizer, f'{self.model_path}/{epoch}/tokenizer')
        torch.save(self.model, f'{self.model_path}/{epoch}/model')

    def load_swot_tokenizer_and_model(self, best_epoch):
        print(f'Trying to load {self.model_type}-{best_epoch} model')

        assert best_epoch < self.epochs, 'best_epoch cannot be more than epochs'

        if not (os.path.exists(f'{self.model_path}/{best_epoch}/model') and \
                os.path.exists(f'{self.model_path}/{best_epoch}/tokenizer')):
            print('No Fine-tuned Model is Found!!!!!!')
            print(f'Fine-tuning {self.model_type} ....it may takes couple of hours...please wait!!')
            self.finetune()
            self.clear_memory(self.model, self.tokenizer)

        print('Loading Saved Models and Tokenizer')
        if self.device.type == 'cpu':
            self.tokenizer = torch.load(
                f'{self.model_path}/{best_epoch}/tokenizer',
                map_location=self.device)
            self.model = torch.load(
                f'{self.model_path}/{best_epoch}/model',
                map_location=self.device)
        else:
            self.tokenizer = torch.load(
                f'{self.model_path}/{best_epoch}/tokenizer')
            self.model = torch.load(
                f'{self.model_path}/{best_epoch}/model')

    def predict(self, crawler_dataloader):
        self.model.eval()
        predictions = []
        sent_ids = []
        for step, batch in tqdm(enumerate(crawler_dataloader),
                                desc='Predicting results....',
                               total = len(crawler_dataloader)):
            batch = tuple(b.to(self.device) for b in batch)
            b_input_ids, b_input_masks, b_sent_ids = batch
            #IMPORTANT
            with torch.no_grad():
                if self.model_type in ['bert', 'roberta','electra']:
                    output = self.model(b_input_ids, attention_mask=b_input_masks, token_type_ids=None)
                elif self.model_type == 'distilbert':
                    output = self.model(b_input_ids, attention_mask=b_input_masks)
            logits = output['logits'].detach().to('cpu').numpy()
            b_sent_ids_cpu = b_sent_ids.to('cpu').numpy()
            predictions.append(logits)
            sent_ids.append(b_sent_ids_cpu)
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_sent_ids = [item for sublist in sent_ids for item in sublist]
        return flat_sent_ids, flat_predictions

    def predict_interface(self, crawler_data):
        '''
        crawler_data = [{'description':'__sentence__', 'sentence_id':_id_}]
        '''
        crawler_df = pd.DataFrame(crawler_data)
        crawler_df = self.preprocessing_on_data(crawler_df)
        crawler_dataloader = self.generate_crawler_dataloader(crawler_df)
        p_results = dict(zip(*self.predict(crawler_dataloader)))
        label_category = {v:k for k,v in self.get_category_lables().items()}
        crawler_df['category'] = crawler_df['sentence_id'].apply(
            lambda sent_id: label_category[p_results[sent_id]])
        return crawler_df[['sentence_id','description','category']]

