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

# How to start for developers:
- update project from Git
- create environment 
```bash
poetry export --without-hashes --format requirements.txt --output requirements.txt
pip install -r requirements.txt
```