# Model Architecture
The pre-trained BERT model was fine-tuned on the classification task using the Trainer class from the transformers library. The model consists of 12 transformer layers, with a hidden size of 768 and 12 attention heads. The model was fine-tuned for 3 epochs with a batch size of 16 for training and 64 for evaluation. The learning rate was warmed up for 500 steps and weight decay of 0.01 was applied. The best model based on accuracy was saved and evaluated on the test set.

