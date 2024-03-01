import sys
env = ["zyf", "huang"]
import os
dir = os.listdir("/ai")
current_env = ""
for e in env:
    if e in dir:
        current_env = e
        break
sys.path.append(f"/ai/{current_env}/project")
from ner.model import BertForTaggingWithCharPooling
from ner import tokenizer
from ner.model import BertForTagging, AdapterForTagging
from ner.trainer import get_optimizer_grouped_parameters, EvaluateCallable
from ner.arguments import Arguments
import torch
from ner.metrics import Metrics
from util.read_file import Reader
from transformers import AutoConfig, default_data_collator, Trainer
import config

if __name__ == "__main__":
    checkpoint_name = "biobert"
    out_putdir = "/models"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 12
    batch_size = 64
    num_train_epochs = 100
    use_datasets = config.datasets_repo[1]
    use_adapter = False
    use_spacy = False
    use_tensorboard = True
    config_json = config.config(current_env=current_env)
    tensorboard_name = use_datasets
    spacy_tag = "spacy" if use_spacy else "no_spacy"
    tensorboard_name = checkpoint_name + "_" + tensorboard_name + "_" + spacy_tag + "_char_mlm"
    checkpoint = f"/ai/{current_env}/trained_models/cdr/mlm_25/checkpoint-9000"
    tensorboard_path = f"/ai/{current_env}/tensorboard_logs"
    adapter_name = tensorboard_name
    if use_adapter:
        tensorboard_name = tensorboard_name + "_" + "adapter"
    tags = config_json["tags"][use_datasets]
    datasets_path = config_json["datasets"][use_datasets]
    # token_cache_path = config["token_cache"][use_datasets]["no_spacy_char"]
    token_cache_path = config.token_cache(config.token_cache_repo[4], use_datasets, current_env)
    label2id = {context: i for i, context in enumerate(tags)}
    id2label = {i: context for i, context in enumerate(tags)}
    reader = Reader(datasets_path)
    datasetDict = reader.get_datasets()
    tokenizer_obj = tokenizer.NERTokenizer(checkpoint=checkpoint, label2id=label2id, id2label=id2label,
                                           datasetDict=datasetDict, token_cache_path=token_cache_path,
                                           use_spacy=use_spacy)
    dataloader = tokenizer_obj.get_data_with_char()
    train_args = Arguments.arguments(output_dir=out_putdir, num_train_epochs=num_train_epochs, batch_size=batch_size,
                                     seed=seed)
    model = BertForTaggingWithCharPooling.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
        checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id))
    # tokenizer.CHARTokenizer(
    #     checkpoint=checkpoint, datasetDict=reader.get_label_words()).put_model_char_embedding(model=model)
    metrics_obj = Metrics(id2label=id2label)
    dataset_len = len(dataloader["train"])
    total_steps = (dataset_len // batch_size) * train_args.num_train_epochs if dataset_len % batch_size == 0 else \
        (dataset_len // batch_size + 1) * train_args.num_train_epochs
    train_args.warmup_steps = 0.1 * total_steps
    metrics = metrics_obj.metrics_for_trainer
    optimizer = None if use_adapter else get_optimizer_grouped_parameters(args=train_args, model=model,
                                                                          num_training_steps=total_steps)
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        train_dataset=dataloader["train"],
        eval_dataset=dataloader["validation"],
        compute_metrics=metrics,
        optimizers=optimizer
    )
    trainer.add_callback(EvaluateCallable(trainer=trainer, use_tensorboard=use_tensorboard, use_adapter=use_adapter,
                                          dataloader=dataloader, tensorboard_name=tensorboard_name,
                                          tensorboard_path=tensorboard_path))
    trainer.train()
