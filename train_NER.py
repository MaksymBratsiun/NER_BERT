import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader


LABELS = ["O", "b-mount", "i-mount"]
PATH_NAMES = "ua_mountains.txt"
PATH_DATA = "ua_text.txt"
PATH_SAVE = "saved_model.save"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    if len(tokenized_sentence) == len(labels):
        return tokenized_sentence, labels


def create_labels_dict(file):
    names_labels = {}
    with open(file, "r", encoding="utf-8") as fd:
        res = fd.readlines()
        res = [i.strip() for i in res]
        res = [i.removesuffix(" \n") for i in res]
        res = [i.removesuffix("\n") for i in res]
    for name in res:
        if len(name.split(" ")) > 1:
            subnames = name.split(" ")
            first_name = {subnames[0].lower(): "b-mount"}
            names_labels.update(first_name)
            second_name = {subnames[i].lower(): "i-mount" for i in range(1, len(subnames[1:]) + 1)}
            names_labels.update(second_name)
        elif len(name.split("-")) > 1:
            subnames = name.split("-")
            names_labels.update({subnames[0].lower(): "b-mount"})
            names_labels.update({subnames[i].lower(): "i-mount" for i in range(1, len(subnames[1:]) + 1)})
        else:
            names_labels.update({name.lower(): "b-mount"})
    return names_labels


def create_data(file):
    with open(file, "r", encoding="utf-8") as fd:
        data_ = []
        while True:
            line = fd.readline()
            if not line:
                break
            data_.append(line.replace("\n", ""))  # yield to generator
    data_ = split_punct(data_, "!")
    data_ = split_punct(data_, "?")
    data_ = split_punct(data_, ".")
    return data_


def tokenized_data(data_input, tokenizer):
    res = []
    for sentence in data_input:
        sentence_tokenized = tokenizer.tokenize(sentence.strip())
        splited_sentence = []
        word = []
        for i in range(len(sentence_tokenized) - 1):
            if i != 0:
                if sentence_tokenized[i].startswith("##"):
                    if not word:
                        word.append(splited_sentence.pop())
                        word.append(sentence_tokenized[i].removeprefix("##"))
                    else:
                        word.append(sentence_tokenized[i].removeprefix("##"))
                else:
                    if word:
                        splited_sentence.append("".join(word))
                        word = []
                        splited_sentence.append(sentence_tokenized[i])
                    else:
                        splited_sentence.append(sentence_tokenized[i])
            else:
                splited_sentence.append(sentence_tokenized[0])

        masked_layer = []
        for word in splited_sentence:
            if word in labels_dict:
                masked_layer.append(labels_dict[word])
            else:
                masked_layer.append("O")
        res.append(tokenize_and_preserve_labels(splited_sentence, masked_layer, tokenizer))
    return res


def split_punct(text, punct):
    result = []
    for i in list(text):
        r = i.strip().split(punct)
        if len(r) > 1:
            for j in range(len(r)):
                if j < len(r) - 1:
                    result.append(r[j] + " " + punct)
                else:
                    if r[j]:  # last part without "punct", if we have "punct" in the end than last part will be ""
                        result.append(r[j])
        else:
            result.append(i)
    return result


def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    print(f"Training epoch: {epoch + 1}")
    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):

        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (include [CLS]&[SEP])
        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS]&[SEP])
            active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [id2label[id_.item()] for id_ in eval_labels]
    predictions = [id2label[id_.item()] for id_ in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    return labels, predictions


class DataSet(Dataset):
    def __init__(self, x, y, tokenizer, max_len):
        self.len = len(x)
        self.X = x
        self.Y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        tokenized_sentence = self.X[index]
        word_labels = self.Y[index]

        # step 2: add special tokens and corresponding labels
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]  # add special tokens
        word_labels = ["O"] + word_labels + ["O"]  # add special tokens

        # step 3: truncating/padding
        maxlen = self.max_len
        if len(tokenized_sentence) > maxlen:  # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            word_labels = word_labels[:maxlen]
        else:  # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
            word_labels = word_labels + ["O" for _ in range(maxlen - len(word_labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in word_labels]

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len


if __name__ == '__main__':

    label2id = {k: v for v, k in enumerate(LABELS)}
    id2label = {v: k for v, k in enumerate(LABELS)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                       num_labels=len(id2label),
                                                       id2label=id2label,
                                                       label2id=label2id)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    labels_dict = create_labels_dict(PATH_NAMES)
    data = create_data(PATH_DATA)
    data = tokenized_data(data, tokenizer)

    X = [data[i][0] for i in range(len(data))]
    Y = [data[i][1] for i in range(len(data))]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    training_set = DataSet(X_train, y_train, tokenizer, MAX_LEN)
    testing_set = DataSet(X_test, y_test, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model.to(device)

    ids = training_set[0]["ids"].unsqueeze(0)
    mask = training_set[0]["mask"].unsqueeze(0)
    targets = training_set[0]["targets"].unsqueeze(0)

    ids = ids.to(device)
    mask = mask.to(device)
    targets = targets.to(device)

    for epoch in range(EPOCHS):
        train(epoch)

    labels, predictions = valid(model, testing_loader)

    torch.save(model, PATH_SAVE)
    print(f"Saved: {PATH_SAVE}")
