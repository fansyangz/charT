import os

def config(current_env):
    return {
      "wandb_name": "[datasets]_[adapter]_[other_tuning]",
      "checkpoint": {
        "chembert": f"{os.path.dirname(current_env)}/pretrain_models/chembert",
        "biobert": f"{os.path.dirname(current_env)}/pretrain_models/biobert",
        "biobert_large": f"{os.path.dirname(current_env)}/pretrain_models/biobert_large",
        "glm3": f"{os.path.dirname(current_env)}/pretrain_models/chatglm3_6b"
      },
      "tags": {
        "chemdner": ["O", "B", "I"],
        "nlm": ["O", "B", "I"],
        "cdr": ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"],
        "chemrxnextractor": ["O", "B-Prod", "I-Prod"]
      },
      "datasets": {
        "chemdner": {
            # "train": f"/ai/{current_env}/project/datasets/chemdner_train.txt",
            # "dev": f"/ai/{current_env}/project/datasets/chemdner_eva.txt",
            # "test": f"/ai/{current_env}/project/datasets/chemdner_test.txt",
            "train": f"{current_env}/datasets/chemdner_train.txt",
            "dev": f"{current_env}/datasets/chemdner_eva.txt",
            "test": f"{current_env}/datasets/chemdner_test.txt"
        },
        "cdr": {
            # "train": f"/ai/{current_env}/project/datasets/cdr_training.txt",
            # "dev": f"/ai/{current_env}/project/datasets/cdr_development.txt",
            # "test": f"/ai/{current_env}/project/datasets/cdr_test.txt",
            "train": f"{current_env}/datasets/cdr_training.txt",
            "dev": f"{current_env}/datasets/cdr_development.txt",
            "test": f"{current_env}/datasets/cdr_test.txt"
        },
        "chemrxnextractor": {
            # "train": f"/ai/{current_env}/project/datasets/train_optim_concat.txt",
            # "dev": f"/ai/{current_env}/project/datasets/dev_optim_concat.txt",
            # "test": f"/ai/{current_env}/project/datasets/test_optim_concat.txt",
            "train": f"{current_env}/datasets/train_optim_concat.txt",
            "dev": f"{current_env}/datasets/dev_optim_concat.txt",
            "test": f"{current_env}/datasets/test_optim_concat.txt",
        },
          "nlm": {
              # "train": f"/ai/{current_env}/project/datasets/train_optim_concat.txt",
              # "dev": f"/ai/{current_env}/project/datasets/dev_optim_concat.txt",
              # "test": f"/ai/{current_env}/project/datasets/test_optim_concat.txt",
              "train": f"{current_env}/datasets/nlm_train.txt",
              "dev": f"{current_env}/datasets/nlm_dev.txt",
              "test": f"{current_env}/datasets/nlm_text.txt",
          }
      },
    }


datasets_repo = ["chemdner", "cdr", "chemrxnextractor", "nlm"]
token_cache_repo = ["spacy", "no_spacy", "entity_char", "no_spacy_char", "spacy_char"]
syntax_repo = [
    {"NOUN": "[unused8]", "VERB": "[unused16]"},
    {"NOUN": "[unused8]"},
    {"VERB": "[unused16]"},
    {"NOUN": "[unused8]", "VERB": "[unused16]", "ADJ": "[unused1]", "ADP": "[unused2]", "ADV": "[unused3]"},
    {"NOUN": "[unused8]", "VERB": "[unused16]", "PRON": "[unused11]", "PROPN": "[unused12]"},
    {"NOUN": "[unused8]", "VERB": "[unused16]", "ADJ": "[unused1]", "ADP": "[unused2]", "ADV": "[unused3]",
     "PRON": "[unused11]", "PROPN": "[unused12]"},
]


def token_cache(name, datasets, current_env, syntax_index=0):
    parent = f"{os.path.dirname(current_env)}/token_cache/{datasets}/{name}"
    if syntax_index != 0:
        parent = f"{parent}/syntax_{syntax_index}"
    # parent = f"/home/kelab/zhouyangfan/token_cache/{datasets}/{name}"
    if not os.path.exists(parent):
        os.makedirs(parent)
    return {
        "train": f"{parent}/train",
        "validation": f"{parent}/validation",
        "test": f"{parent}/test",
    }