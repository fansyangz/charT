import sys
import os
current_env = "/ai/zyf/charT"
sys.path.append(current_env)
os.system(f"pip install -r {current_env}/requirements.txt")
os.system(f"pip install -r {current_env}/en_core_web_sm-3.7.0-py3-none-any.whl")
from ner import tokenizer
from util.read_file import Reader
import config as config_json
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("train_args")
    group.add_argument("--checkpoint_name", default="biobert")
    group.add_argument("--num_train_epochs", type=int, default=200)
    group.add_argument("--datasets_index", type=int, default=0)
    group.add_argument("--mlm_probability", type=float, default=0.25)
    group.add_argument("--eval_steps", type=int, required=True)
    group.add_argument("--save_steps", type=int, required=True)
    args = parser.parse_args()
    checkpoint_name = args.checkpoint_name
    seed = 12
    batch_size = 64
    num_train_epochs = args.num_train_epochs
    mlm_probability = args.mlm_probability
    config = config_json.config(current_env=current_env)
    datasets_repo = ["chemdner", "cdr", "chemrxnextractor"]
    use_datasets = datasets_repo[args.datasets_index]
    checkpoint = config["checkpoint"][checkpoint_name]
    tags = config["tags"][use_datasets]
    datasets_path = config["datasets"][use_datasets]
    output_dir = rf"/ai/zyf/trained_model/{use_datasets}/{str(mlm_probability)}"
    # token_cache_path = config["token_cache"][use_datasets]["entity_char"]
    label2id = {context: i for i, context in enumerate(tags)}
    id2label = {i: context for i, context in enumerate(tags)}
    label_words = Reader(datasets_path).get_label_words()
    mlm_model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    tokenizer_obj = tokenizer.CHARTokenizer(checkpoint=checkpoint, datasetDict=label_words)
    datesets, data_collator = tokenizer_obj.datasets_char_map(mlm_probability=mlm_probability)
    mlm_train_args = TrainingArguments(
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
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps
    )
    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_train_args,
        tokenizer=tokenizer_obj.tokenizer,
        data_collator=data_collator,
        train_dataset=datesets["train"],
        eval_dataset=datesets["test"],
    )
    mlm_trainer.train()


