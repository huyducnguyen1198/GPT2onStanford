import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader,Dataset

# load model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
''' for preprocessing and tokenizing data 
dataset = load_dataset("hugfaceguy0001/stanford_plato")

training = dataset["train"]

###############################
# get all text from batch     #
###############################
def getText(batch):
    
    get all text from batch
    :param batch:
    :return:
    
    text = []
    for c in batch["main_text"]:
        if c:
            for p in c:
                text.extend(p['main_content'])
    return text
training = getText(training)

###############################
# analyse the text length
# by ploting                    #
###############################
def plotLength(training):
    
    analyse the text length
    by ploting
    :param training:
    :return:
    
    leng = [len(t) for t in training]

    # plot length distribution
    import matplotlib.pyplot as plt
    import numpy as np

    max_length = max(leng)
    bins = list(range(0, max_length + 250, 250))  # +1000 to include the last range

    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
    plt.hist(leng, bins=bins, edgecolor='k', alpha=0.7)  # edgecolor for bin definition
    plt.title('Distribution of Text Lengths by Thousands')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.xticks(bins)  # Set x-ticks to clearly show the thousand ranges

    plt.show()
#plotLength(training)

###############################
# divide the text into bins    #
# and cut and concat           #
# to get the same length  1000 #
###############################
def processText(dataset):
    
    aranage the text into chunks of 500-1000
    :param dataset: an array of text varies in length
    :return:  an array of text chunks of 500-1000

    # Define step size
    step = 500
    # Count numbers in each interval of 250
    textInBins = [[ number for number in dataset if lower < len(number) <= upper] for lower, upper in
              zip(range(0, 3000, step), range(step, 3001, step))]

    def breakDown(text):

        break down the text into interval
        :param text:
        :param interval:
        :return:

        if len(text) < step:
            return [text]
        return [text[i:i + step] for i in range(0, len(text), step)]
    def concat(c1, c2):

        concat the text to get the same length
        :param text:
        :return:

        return c1 + c2

    chunkList = [chunk for text in textInBins for para in text for chunk in breakDown(para)]
    chunkList = [concat(c1, c2) for c1, c2 in zip(chunkList, chunkList[1:])]

    return chunkList

training = processText(training)



def tokenize_function(dataset, tokenizer):

    tokenize the text
    :param dataset: an array of text chunks of 500-1000
    :param tokenizer:
    :return:

    input_ids = ['']*len(dataset)
    attention_mask = ['']*len(dataset)
    tokensText = ['']*len(dataset)

    for i, text in enumerate(dataset):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokensText[i] = encoded

    return tokensText

#trainingSet = tokenize_function(training[:100], tokenizer)
trainingSet = tokenizer(training, padding=True, truncation=True, max_length=1024, return_tensors="pt")
torch.save(trainingSet, "tokenized_texts.pt")

print("finish tokenizing")
print(trainingSet)
exit()'''

# load tokenized text
trainingSet = torch.load("tokenized_texts.pt")


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()} #torch.tensor(val[idx])
        item["labels"] = item["input_ids"].clone()  # Labels are required for training
        return item

    def __len__(self):
        return self.encodings.input_ids.size(0)

# Create the dataset
dataset = TextDataset(trainingSet)
#dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Adjust batch_size as needed
#torch.device("mps")

# load model
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-stanford-plato",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_total_limit=2,
    eval_steps=100,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #compute_metrics=compute_metrics,
)

print("start training")
trainer.train()

trainer.save_model("./gpt2-stanford-plato")
