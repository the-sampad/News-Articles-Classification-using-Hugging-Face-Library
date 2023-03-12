import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TextDataset

# load the dataset into a pandas dataframe
df = pd.read_csv('bbc-text.csv')

# preprocess the text data
def preprocess_text(text):
    # convert to lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if not token in stop_words]
    # rejoin the tokens into a string
    text = ' '.join(tokens)
    return text

# apply the preprocess_text function to the 'text' column of the dataframe
df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into training, validation, and test sets. 
# Using the train_test_split function from scikit-learn to do this
train_data, test_data, train_labels, test_labels = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# load a pre-trained BERT or GPT-2 model
model_name = 'bert-base-uncased' # or 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5) # the number of labels is 5 for the 5 categories in the dataset

# Tokenize the text data using the tokenizer and encode the labels as integers
train_encodings = tokenizer(train_data.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_data.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data.tolist(), truncation=True, padding=True)

train_labels_enc = [train_labels.cat.categories.get_loc(label) for label in train_labels]
val_labels_enc = [val_labels.cat.categories.get_loc(label) for label in val_labels]
test_labels_enc = [test_labels.cat.categories.get_loc(label) for label in test_labels]

# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='steps',     # evaluate every eval_steps
    eval_steps=50,                   # evaluation steps
    load_best_model_at_end=True,     # load the best model at the end of training
    metric_for_best_model='accuracy',# use accuracy to determine the best model
    greater_is_better=True           # higher metric value is better
)

# defining the train, test and evaluation datasets
train_dataset = TextDataset.from_dict({'input_ids': train_encodings['input_ids'],
                                    'attention_mask': train_encodings['attention_mask'],
                                    'labels': train_labels_enc})

eval_dataset = TextDataset.from_dict({'input_ids': val_encodings['input_ids'],
                                    'attention_mask': val_encodings['attention_mask'],
                                    'labels': val_labels_enc})

test_dataset = TextDataset.from_dict({'input_ids': test_encodings['input_ids'],
                                    'attention_mask': test_encodings['attention_mask'],
                                    'labels': test_labels_enc})

# defining the trainer
trainer = Trainer(
    model=model,                     # the instantiated Transformers model to be trained
    args=training_args,              # training arguments, defined above
    train_dataset=train_dataset,     # training dataset
    eval_dataset=eval_dataset        # evaluation dataset
)

# fine-tune the model
trainer.train()



# evaluating the model on the test set
results = trainer.evaluate(test_dataset)

# print the evaluation results
print(results)


