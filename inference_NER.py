import torch
from transformers import BertTokenizer


LABELS = ["O", "b-mount", "i-mount"]
PATH_SAVE = "saved_model.save"
MAX_LEN = 128


class BColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_C = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


COLOR_MODE_ON = BColors.HEADER
COLOR_MODE_OFF = BColors.END_C


def predict_to_print(words_predict):

    splited_sentence = []
    word = []

    for pair in words_predict:
        if pair[0] in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        elif pair[0].startswith("##"):
            if not word:
                if pair[1] != "O":
                    word.append(COLOR_MODE_ON + splited_sentence.pop() + COLOR_MODE_OFF)
                    word.append(COLOR_MODE_ON + pair[0].removeprefix("##") + COLOR_MODE_OFF)
                else:
                    word.append(splited_sentence.pop())
                    word.append(pair[0].removeprefix("##"))

            else:
                if pair[1] != "O":
                    word.append(COLOR_MODE_ON + pair[0].removeprefix("##") + COLOR_MODE_OFF)
                else:
                    word.append(pair[0].removeprefix("##"))
        else:
            if word:
                if pair[1] != "O":
                    splited_sentence.append("".join(word))
                    word = []
                    splited_sentence.append(COLOR_MODE_ON + pair[0] + COLOR_MODE_OFF)
                else:
                    splited_sentence.append("".join(word))
                    word = []
                    splited_sentence.append(pair[0])
            else:
                if not splited_sentence:
                    word_in_sentence = pair[0].title()
                else:
                    word_in_sentence = pair[0]

                if pair[1] != "O":
                    splited_sentence.append(COLOR_MODE_ON + word_in_sentence.title() + COLOR_MODE_OFF)
                else:
                    splited_sentence.append(word_in_sentence)
    return " ".join(splited_sentence)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = torch.load(PATH_SAVE)
    model.eval()

    label2id = {k: v for v, k in enumerate(LABELS)}
    id2label = {v: k for v, k in enumerate(LABELS)}

    sentence = input("Enter the sentence: ")

    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

    # move to device
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)

    # forward pass
    outputs = model(ids, mask)
    logits = outputs[0]

    # shape (batch_size * seq_len, num_labels)
    active_logits = logits.view(-1, model.num_labels)

    # shape (batch_size*seq_len,) - predictions at the token level
    flattened_predictions = torch.argmax(active_logits, axis=1)

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    print(predict_to_print(wp_preds))
