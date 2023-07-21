'''
!pip install transformers
!pip install sentence_transformers
!pip install torchmetrics
!pip install openai
'''

# importing
import sys

import numpy as np  # to handle data
import pandas as pd  # to handle and save data
import os
import pickle
from datetime import datetime as d  # to generate timestamps to save models
import math
import random

# from sentence_transformers import SentenceTransformer  # for word embedding
from transformers import AutoTokenizer, AutoModel

import torch  # for AI
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
from torchmetrics import R2Score

import matplotlib.pyplot as plt  # to plot training
import openai  # to generate training data

import seaborn as sns  # to analyse data
from wordcloud import WordCloud

# import nltk
# nltk.download('stopwords')  # uncomment this line to use the NLTK Downloader
from nltk.corpus import stopwords

openai.api_key = os.getenv('OPENAI_API_KEY')


class CustomMultiClassDataset(Dataset):  # the dataset class
    def __init__(self, sentences, labels):
        self.x = sentences
        self.y = labels
        self.length = self.x.shape[0]
        self.shape = self.x[0].shape[0]
        self.feature_names = ['sentences', 'labels']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NeuralNetwork(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack_with_sigmoid = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack_with_sigmoid(x)
        return logits


class History:  # The history object to keep track of metrics during training and plot graphs to it.
    def __init__(self, val_set, train_set, model, **kwargs):  # kwargs are the metrics to keep track of.
        self.val_set = val_set
        self.train_set = train_set
        self.model = model
        self.kwargs = kwargs
        self.history = {'steps': []}
        for i in kwargs.keys():
            self.history.update({'val_' + i: []})
            self.history.update({'tra_' + i: []})
        self.valloader = None
        self.trainloader = None

    def save(self, step):  # this function is called in the training loop to save the current state of the model.
        short_history = {}
        for i in self.kwargs.keys():
            short_history.update({'val_' + i: []})
            short_history.update({'tra_' + i: []})
        # generate two dataloader with each k entries from either the training or the validation data.
        k = 500
        short_train_set, waste = torch.utils.data.random_split(self.train_set, [k, len(self.train_set) - k])
        short_val_set, waste = torch.utils.data.random_split(self.val_set, [k, len(self.val_set) - k])
        self.valloader = DataLoader(dataset=short_val_set, batch_size=5, shuffle=True, num_workers=2)
        self.trainloader = DataLoader(dataset=short_train_set, batch_size=5, shuffle=True, num_workers=2)
        # iterate over both dataloaders simultaneously
        for i, ((val_in, val_label), (tra_in, tra_label)) in enumerate(zip(self.valloader, self.trainloader)):
            with torch.no_grad():
                self.model.eval()
                # predict outcomes for training and validation.
                val_pred = self.model(val_in)
                tra_pred = self.model(tra_in)
                for j in self.kwargs.keys():  # iterate over the metrics
                    # calculate metric and save to short history
                    if len(val_pred) > 1:
                        val_l = self.kwargs[j](val_pred, val_label).item()
                        tra_l = self.kwargs[j](tra_pred, tra_label).item()
                        short_history['val_' + j].append(val_l)
                        short_history['tra_' + j].append(tra_l)
                self.model.train()
        # iterate over metrics and save the average of the short history to the history.
        for i in self.kwargs.keys():
            self.history['val_' + i].append(sum(short_history['val_' + i]) / len(short_history['val_' + i]))
            self.history['tra_' + i].append(sum(short_history['tra_' + i]) / len(short_history['tra_' + i]))
        self.history['steps'].append(step)  # save steps for the x-axis

    # this function is called after training to generate graphs.
    # When path is given, the graphs are saved and plt.show() is not called.
    def plot(self, path=None):
        figures = []
        for i in self.kwargs.keys():  # iterate over the metrics and generate graphs for each.
            fig, ax = plt.subplots()
            ax.plot(self.history['steps'], self.history['val_' + i], 'b')
            ax.plot(self.history['steps'], self.history['tra_' + i], 'r')
            ax.set_title(i.upper())
            ax.set_ylabel(i)
            ax.set_xlabel('Epochs')
            figures.append(fig)
            if path is None:
                plt.show()  # depending on the setup the graphs might still be shown even without this function called.
            else:
                plt.savefig(f"{path}/{i}")
            plt.clf()
        return figures


# this function is copied from https://huggingface.co/sentence-transformers/all-roberta-large-v1
# it returns embedded versions of the sentences its passed.
def long_roberta(sentences):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # test if this works with truncation=False

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


# This function embeds the data and saves it to f'embedded_data_{topic}.pt'.
# This function can take a while therefore it saves all embedded data every k=100 sentences in case of an error.
def prepare_data_slowly(data, name):
    np_data = data.to_numpy().transpose()
    # the used model in long_roberta is SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    embedded_data = np.array([[0, 0]])
    k = 100
    for i in range(math.ceil(len(np_data[0]) / k)):
        sentences = long_roberta(list(np_data[0][k * i:k * i + k]))  # embed the sentences
        labels = np_data[1][k * i:k * i + k]
        a = np.array([torch.tensor_split(sentences, len(sentences)), labels])
        a = a.transpose()
        embedded_data = np.append(embedded_data, a, axis=0)
        if i == 0:
            embedded_data = embedded_data[1:]
        torch.save(embedded_data, f'embedded_data_{name}.pt')
        print(f'saved {i + 1} / {len(np_data[0]) / k}')
    return embedded_data.transpose()


# check whether the input data is short enough for word embedding.
def check_length(data):
    def tokenize(sentences):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
        encoded_input = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
        return encoded_input

    sorted_data = data.reindex(data.sentences.str.len().sort_values().index[::-1]).reset_index(drop=True)

    for idx, row in sorted_data.iterrows():
        length = len(tokenize(row.sentences)['input_ids'][0])
        if length > 512:
            print('Warning: Paragraph longer than 512 tokens therefore it is too long and will be truncated.')
        elif length > 128:
            print('Warning: Paragraph longer than 128 tokens therefore longer than recommended.')
        elif length < 80:
            break  # as the data is sorted by (string) length there shouldn't be any problems after this point.


# analyse the data (balance, wordcloud relative to label, input length relative to label, duplicates of inputs and
# null inputs or labels)
def analyse_full_data(data):
    print('INFO')
    data.info()
    data.groupby(['labels']).describe()
    print(f'Number of unique sentences: {data["sentences"].nunique()}')
    duplicates = data[data.duplicated()]
    print(f'Number of duplicate rows:\n{len(duplicates)}')
    print(f'Check for nulls:\n{data.isnull().sum()}')
    sns.countplot(x=data['labels'])  # ploting distribution for easier understanding
    print(data.head(3))

    print('A few random examples from the dataset:')
    # let's see how data is looklike
    random_index = random.randint(0, data.shape[0] - 3)
    for row in data[['sentences', 'labels']][random_index:random_index + 3].itertuples():
        _, text, label = row
        print(f'TEXT: {text}')
        print(f'LABEL: {label}')

    def wordcloud_draw(data, color, s):
        words = ' '.join(data)
        cleaned_word = " ".join([word for word in words.split() if (word != 'movie' and word != 'film')])
        wordcloud = WordCloud(stopwords=stopwords.words('english'), background_color=color, width=2500,
                              height=2000).generate(cleaned_word)
        plt.imshow(wordcloud)
        plt.title(s)
        plt.axis('off')

    plt.figure(figsize=[20, 10*data["labels"].nunique()])  # TODO: Check whether to adjust the figure size
    for i in range(data["labels"].nunique()):
        one_hot = np.zeros(data["labels"].nunique())
        one_hot[i] = 1
        draw_data = pd.DataFrame()
        for j in data.itertuples():
            if j['labels'] == one_hot:
                draw_data.append(j)
        # draw_data = data[data['labels'] == one_hot]
        # draw_data = draw_data['sentences']
        plt.subplot(1, 2, i)
        wordcloud_draw(draw_data, 'white', f'Most-common words in category {i}.')

    plt.show()  # end wordcloud

    data['text_word_count'] = data['sentences'].apply(lambda x: len(x.split()))

    numerical_feature_cols = ['text_word_count']  # numerical_feature_cols = data['text_word_count']

    plt.figure(figsize=(20, 3))
    for i, col in enumerate(numerical_feature_cols):
        plt.subplot(1, 3, i + 1)
        sns.histplot(data=data, x=col, bins=50, color='#6495ED')
        plt.title(f"Distribution of Various word counts")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 3))
    for i, col in enumerate(numerical_feature_cols):
        plt.subplot(1, 3, i + 1)
        sns.histplot(data=data, x=col, hue='labels', bins=50)
        plt.title(f"Distribution of Various word counts with respect to target")
    plt.tight_layout()
    plt.show()


class DYNAMIC_AI:
    def __init__(self, name):
        self.name = name
        self.running_loss = None
        self.optimizer = None
        self.dataloader = None
        self.model = None
        self.loss = None
        self.dataframe = None
        self.val_set = None
        self.train_set = None
        self.labels = None
        self.sentences = None
        self.embedded_data = None
        self.raw_data = None
        self.dataset = None

    # generates training data (generate_data()) if real=False loads data instead
    def generate_training_data(self, prompt_nr=100, answer_nr=100, *prompts):
        def ask_ai(prompt):  # get nr of answers from a prompt. Prompt should end with '\n\n1.'.
            response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1,
                                                max_tokens=10 * answer_nr)
            response = '1.' + response['choices'][0]['text'] + '\n'
            li = []
            for i in range(answer_nr):
                pos = response.find(str(i + 1))
                beg = pos + len(str(i + 1)) + 2
                end = response[beg:].find('\n')
                li.append(response[beg:beg + end])
            return li

        def gen_sentences():  # generates nr keywords to the prompt and 50*factor sentences to each
            prompt_variations = ask_ai(master_prompt)
            sentences = []
            for i in prompt_variations:
                sentences.extend(ask_ai(prompt=f'{i}\n\n1.'))
            return sentences

        all_sentences = []
        labels = []
        max = len(prompts)
        for idx, i in enumerate(prompts):
            master_prompt = f'Give me {prompt_nr} variations of this prompt: "{i}".\n\n1.'
            new_sentences = gen_sentences()
            for j in new_sentences:
                if j not in all_sentences:
                    all_sentences.append(j)
                    one_hot = np.zeros(max)
                    one_hot[idx] = 1
                    labels.append(one_hot)

        data = [all_sentences, labels]
        data = np.array(data).transpose()
        '''
        mapping = []
        uni = np.unique(data)
        for i in uni:
            mapping.append(np.where(data == i)[0][0])
        data = data[mapping[1:]]
        '''
        pd.DataFrame(data).to_csv(f"{self.name.replace(' ', '_')}_generated_data.csv", index=False,
                                  header=['sentences', 'labels'])
        self.raw_data = pd.read_csv(f"{self.name.replace(' ', '_')}_generated_data.csv")

    # embeds the raw_data and creates a dataset from it
    def embed_data(self, real=True):
        def get_element(arr):
            return arr[0]

        if real:
            self.embedded_data = prepare_data_slowly(self.raw_data, self.name)
        else:
            self.embedded_data = torch.load(f'embedded_data_{self.name}.pt').transpose()
        # convert embedded data to torch dataset
        tpl = tuple(map(get_element, tuple(np.array_split(self.embedded_data[0], len(self.embedded_data[0])))))
        self.sentences = torch.cat(tpl)
        self.labels = self.embedded_data[1]
        # self.labels[self.labels is True] = 1.
        # self.labels[self.labels is False] = 0.
        self.labels = np.expand_dims(self.labels, axis=1).astype('float32')  # TODO: Check this line for multi choice
        self.labels = torch.from_numpy(self.labels)
        self.dataset = CustomMultiClassDataset(self.sentences, self.labels)

    # calls analyse_full_data with the raw_data
    def analyse_training_data(self):
        # check_length(self.raw_data)
        analyse_full_data(self.raw_data)

    # this function trains a model and returns it as well as the history object of its training process.
    def train(self, epochs=10, lr=0.001, val_frac=0.1, batch_size=25, loss=nn.CrossEntropyLoss()):
        # get_acc measures the accuracy and is passed as a metric to the history object.
        def get_acc(pred, target):
            pred_tag = torch.round(pred)  # TODO: Does this work with multi-class?

            correct_results_sum = (pred_tag == target).sum().float()
            acc = correct_results_sum / target.shape[0]
            acc = torch.round(acc * 100)

            return acc

        # generate validation dataset with the fraction of entries of the full set passed as val_frac
        val_len = int(round(len(self.dataset) * val_frac))
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset,
                                                                     [len(self.dataset) - val_len, val_len])
        self.dataloader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        self.model = NeuralNetwork(self.dataset.shape, self.dataset[0].labels.shape)

        self.loss = loss  # the loss passed to this train function
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # define metrics to be monitored by the history object during training.
        mseloss = nn.MSELoss()
        accuracy = get_acc

        # define history object
        history = History(self.val_set, self.train_set, self.model, mseloss=mseloss, accuracy=accuracy)

        # main training loop
        for epoch in range(epochs):
            self.running_loss = 0.
            print(f'Starting new batch {epoch + 1}/{epochs}')
            for step, (inputs, labels) in enumerate(self.dataloader):
                y_pred = self.model(inputs)
                lo = self.loss(y_pred, labels)
                lo.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.running_loss += lo.item()
                if (step + 1) % math.floor(len(self.dataloader) / 5 + 2) == 0:  # if (step+1) % 100 == 0:
                    print(f'current loss:\t\t{self.running_loss / 100}')
                    self.running_loss = 0
                    history.save(epoch + step / len(self.dataloader))
                    # save current state of the model to history
        # generate folder with timestamp and save the model there.
        now = str(d.now().isoformat()).replace(':', 'I').replace('.', 'i').replace('-', '_')
        os.mkdir(f"model_{now}")
        torch.save(self.model, f"model_{now}/model.pt")
        print(f'Model saved to "model_{now}/model.pt"')
        history.plot(f"model_{now}")  # save graphs to the folder
        return history, self.model  # return history and model


# with this function you can pass custom sentences to the model
def try_model(model):
    a = input('Please enter your input sentence: ')
    a = long_roberta(a)
    pred = model(a)
    print(pred.item())
    print('Where 1 is the first prompt: ""\nand 0 is the second: "".\n')


if __name__ == "__main__":
    ti = DYNAMIC_AI('subject_classifier')
    biology_prompt = 'Write a sentence about biology.'
    chemistry_prompt = 'Write a sentence about chemistry.'
    physics_prompt = 'Write a sentence about physics.'
    geology_prompt = 'Write a sentence about geology.'
    ti.generate_training_data(2, 3, biology_prompt, chemistry_prompt, physics_prompt, geology_prompt)
    print('ANALYSE DATA BEGINN')
    ti.analyse_training_data()
    print('ANALYSE DATA END')
    ti.embed_data()
    history, model = ti.train(epochs=10, lr=0.0001, val_frac=0.1, batch_size=10, loss=nn.BCELoss())
    history.plot()


# Far higher diversity in not topic related samples needed. Normal conversation, random sequences of letters, etc.
