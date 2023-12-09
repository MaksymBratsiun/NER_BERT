# NER_BERT
# Task:
Natural Language Processing.
Named entity recognition.

In this task, we need to train a named entity recognition (NER) model for the identification of
mountain names inside the texts. For this purpose you need:

- Find / create a dataset with labeled mountains.
- Select the relevant architecture of the model for NER solving.
- Train / finetune the model.
- Prepare demo code / notebook of the inference results.

The output for this task should contain:

- Jupyter notebook that explains the process of the dataset creation.
- Dataset including all artifacts it consists of.
- Link to model weights.
- Python script (.py) for model training.
- Python script (.py) for model inference.
- Jupyter notebook with demo.

# Overview
This repository contains Python scripts for training and inference of a Named Entity Recognition (NER) model based 
on BERT (Bidirectional Encoder Representations from Transformers). The training script (train_NER.py) trains the model
on a custom dataset and saves the trained model. The inference script (inference_NER.py) loads the trained model
and performs predictions on new input text.

# Requirements
- Python 3.x
- PyTorch
- Transformers 
- Other dependencies (install using "pip install -r requirements.txt")

# Usage:
- update project from Git
- create virtual environment 
```bash
pip install -r requirements.txt
```
Run train_NER.py for create and training model
```bash
python  train_NER.py
```
Run inference_NER.py for input test sentence. Its highlight mountains name.
```bash
python inference_NER.py
```
Input sentence for test. Press Enter

Adjust the script parameters:
- LABELS = ["O", "b-mount", "i-mount"]
- PATH_NAMES = "ua_mountains.txt"
- PATH_DATA = "ua_text.txt"
- PATH_SAVE = "saved_model.save"
- MAX_LEN = 128
- TRAIN_BATCH_SIZE = 4 
- VALID_BATCH_SIZE = 2
- EPOCHS = 5
- LEARNING_RATE = 1e-05
- MAX_GRAD_NORM = 10


Modify hyperparameters and model configurations in the training script.
Customize the inference script for your specific use case.

# Conclusion
Pretrained BERT have easy implementation, it save our time for pretraining and have good accuracy.
With default parameters we have result:
- Training epoch: 5
- Training loss epoch: 0.040255760230744876
- Training accuracy epoch: 0.9620963169021954
- Validation Loss: 0.015128238980347911
- Validation Accuracy: 0.9966666666666667
