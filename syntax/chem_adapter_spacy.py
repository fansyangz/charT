# dataloader
from datasets import Dataset, DatasetDict
import syntax
from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np
from transformers.adapters import AutoAdapterModel
from transformers.adapters import AdapterTrainer
from transformers import default_data_collator
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, TrainingArguments, EvalPrediction
# tokenize
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import set_seed
from transformers.trainer_callback import DefaultFlowCallback, TrainerState, TrainerControl
import wandb

use_wandb = True
if use_wandb:
    wandb.login()
if use_wandb:
    wandb.init(project="chemdner_adapter_spacy")

# 获取数据集
# train = "train_optim_concat.txt"
# dev = "dev_optim_concat.txt"
# test = "test_optim_concat.txt"
train = "../chemdner_train.txt"
dev = "../chemdner_eva.txt"
test = "../chemdner_test.txt"


checkpoint = "/ai/zyf/chembert"
# checkpoint = "/chembert/models"
output_dir = "/models"
bert_tag_label_id = -100
spacy_tag_label_id = -100
pad_label_id = -100
pad_token = 0
# prod_labels = ["O", "B-Prod", "I-Prod"]
prod_labels = ["O", "B", "I"]

label2id = {context: i for i, context in enumerate(prod_labels)}
id2label = {i: context for i, context in enumerate(prod_labels)}


def read_examples_from_file(file_path):
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words, labels = [], []
        for line in f:
            line = line.rstrip()
            if line == "":
                if words:
                    examples.append({
                        "words": words,
                        "labels": labels
                    })
                    words, labels = [], []
            else:
                splits = line.split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1])
                else:
                    # Examples could have no label for plain test files
                    labels.append("O")
        if words:
            examples.append({
                "words": words,
                "labels": labels
            })

    return examples


train_input = read_examples_from_file(train)
dev_input = read_examples_from_file(dev)
test_input = read_examples_from_file(test)
# just useing the  noun and verb
spacy_pos_token = {"NOUN": "[unused8]", "VERB": "[unused16]", }

CLS = "[CLS]"
SEP = "[SEP]"
max_sentence = 256
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

token_cache_path = {
    "train": "/ai/zyf/token_cache/token_cache_train_no_spacy_chemdner",
    "validation": "/ai/zyf/token_cache/token_cache_validation_no_sapcy_chemdner",
    "test": "/ai/zyf/token_cache/token_cache_test_no_sapcy_chemdner"
}


def build_spacy(sent_list):
    spacy_model = spacy.load("en_core_web_sm")
    # en = English()
    return {
        "syntax": spacy_model(' '.join(sent_list)),
        "find_idx": build_sent_idx(sent_list)
    }


def build_sent_idx(sent_list):
    total_len = 0
    find_idx = []
    for i, word in enumerate(sent_list):
        find_idx.append(str(total_len) + "-" + str(total_len + len(word) - 1))
        # stridx2listidx[str(total_len) + "-" + str(total_len + len(word))] = i
        total_len = total_len + len(word) + 1
    return find_idx


def do_find_idx(find_idx, idx):
    for i, func in enumerate(find_idx):
        func_list = func.split("-")
        if (int(func_list[0]) <= idx and int(func_list[1]) >= idx):
            return i


def check_append_word(tokens_list, word):
    if (len(tokens_list) + 1) > max_sentence - 1:
        return
    tokens_list.append(word)


def tokenize_function(example):
    word_list = example["words"]
    original_labels = example["labels"]
    spacy_result = build_spacy(word_list)
    spacy = spacy_result["syntax"]
    sent_idx = spacy_result["find_idx"]
    tokens = [CLS]
    labels = [bert_tag_label_id]
    last_idx = 0
    sub_token_tag = None
    for _, word in enumerate(spacy):
        original_idx = do_find_idx(sent_idx, word.idx)
        # the sub token from same token
        if original_idx == last_idx:
            if not sub_token_tag and spacy_pos_token.get(word.pos_):
                sub_token_tag = spacy_pos_token.get(word.pos_)
        # the sub token from new token
        else:
            last_token = word_list[last_idx]
            first_sub_token = True
            for sub_token in tokenizer.tokenize(last_token):
                check_append_word(tokens, sub_token)
                label_tag = original_labels[last_idx] if first_sub_token else "O"
                check_append_word(labels, label2id[label_tag])
                first_sub_token = False
            # add verb or noun token
            if sub_token_tag:
                check_append_word(tokens, sub_token_tag)
                check_append_word(labels, spacy_tag_label_id)
            last_idx = original_idx
    assert len(tokens) == len(labels)
    input_ids, token_type_ids, attention_mask, labels = padding_fn(
        tokenizer.convert_tokens_to_ids(tokens),
        [0] * len(tokens),
        [1] * len(tokens),
        labels,
    )
    decoder_mask = [(x != pad_label_id) for x in labels]
    assert len(input_ids) == max_sentence
    assert len(token_type_ids) == max_sentence
    assert len(attention_mask) == max_sentence
    assert len(labels) == max_sentence
    assert len(decoder_mask) == max_sentence
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "label_ids": labels,
        "decoder_mask": decoder_mask
    }


def tokenize_function_no_spacy(example):
    tokens = []
    label_ids = []
    word_list = example["words"]
    original_labels = example["labels"]

    for word, label in zip(word_list, original_labels):
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            label_ids.extend([label2id[label]] + [pad_label_id] * (len(word_tokens) - 1))

    if len(tokens) > max_sentence - 2:
        tokens = tokens[:(max_sentence - 2)]
        label_ids = label_ids[:(max_sentence - 2)]

    tokens += [SEP]
    label_ids += [pad_label_id]

    tokens = [CLS] + tokens
    label_ids = [pad_label_id] + label_ids

    input_ids, token_type_ids, attention_mask, labels = padding_fn(
        tokenizer.convert_tokens_to_ids(tokens),
        [0] * len(tokens),
        [1] * len(tokens),
        label_ids,
    )
    decoder_mask = [(x != pad_label_id) for x in label_ids]
    assert len(input_ids) == max_sentence
    assert len(token_type_ids) == max_sentence
    assert len(attention_mask) == max_sentence
    assert len(label_ids) == max_sentence
    assert len(decoder_mask) == max_sentence
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "decoder_mask": decoder_mask
    }


seed = 12


set_seed(seed)
def padding_fn(input_ids, token_type_ids, attention_mask, labels):
    pad_len = max_sentence - len(input_ids)
    input_ids += [pad_token] * pad_len
    token_type_ids += [0] * pad_len
    attention_mask += [0] * pad_len
    labels += [pad_label_id] * pad_len
    return input_ids, token_type_ids, attention_mask, labels


config = AutoConfig.from_pretrained(
    checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)
hidden_size = 768
num_train_epochs = 50
train_dataset = Dataset.from_list(train_input)
dev_dataset = Dataset.from_list(dev_input)
test_dataset = Dataset.from_list(test_input)

datasetDict = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset,
    "test": test_dataset
})

tokenized_datasets = datasetDict.map(function=tokenize_function,
                                     cache_file_names=token_cache_path,
                                     load_from_cache_file=True)


standard_columns = ["input_ids", "token_type_ids", "attention_mask", "label_ids", "decoder_mask"]


tokenized_datasets = tokenized_datasets.select_columns(standard_columns)
tokenized_datasets.set_format("torch")


class FocalLossSelf(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=2, alpha=None, ignore_index=-100):
        super().__init__(weight=alpha, ignore_index=ignore_index)
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 32
adapter_name = "chemadapter"

model = AutoAdapterModel.from_pretrained(checkpoint)
model.add_adapter(adapter_name)
model.train_adapter(adapter_name)
model.add_tagging_head(adapter_name, num_labels=config.num_labels)
model.set_active_adapters(adapter_name)

# model.to(device)
train_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    remove_unused_columns=False,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True,
    do_eval=True,
    do_predict=True,
    seed=seed,
    report_to=["none"],
    save_strategy="no"
)


# def ner_compute_metrics(predictions, label_ids):
def ner_compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    label_ids = [[l for l in label if l != -100] for label in labels]
    predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(preds, labels)]
    label_list = [[id2label[x] for x in seq] for seq in label_ids]
    preds_list = [[id2label[x] for x in seq] for seq in predictions]
    return {
        "precision": precision_score(label_list, preds_list),
        "recall": recall_score(label_list, preds_list),
        "f1": f1_score(label_list, preds_list),
    }


trainer = AdapterTrainer(
    model=model,
    args=train_args,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=ner_compute_metrics,
)


class EvaluateCallable(DefaultFlowCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        evaluate = self.trainer.evaluate()
        print("-------------epoch: " + str(state.epoch) + "-------------")
        print("-------------evaluate: " + str(state.epoch) + "-------------")
        print(evaluate["eval_precision"], evaluate["eval_recall"], evaluate["eval_f1"])
        test = self.trainer.predict(tokenized_datasets["test"])
        print("-------------test: " + str(state.epoch) + "-------------")
        print(test.metrics["test_precision"], test.metrics["test_recall"], test.metrics["test_f1"])
        if use_wandb:
            wandb.log({
                "eval_precision": evaluate["eval_precision"],
                "eval_recall": evaluate["eval_recall"],
                "eval_f1": evaluate["eval_f1"],
                # "eval_loss": evaluate["loss"],
                "test_precision": test.metrics["test_precision"],
                "test_recall": test.metrics["test_recall"],
                "test_f1": test.metrics["test_f1"],
                # "test_loss": test["loss"],
            })


trainer.add_callback(EvaluateCallable(trainer=trainer))
trainer.train()

