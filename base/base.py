import sys
import os
py_file_path = os.path.abspath(sys.argv[0])
current_env = os.path.dirname(os.path.dirname(py_file_path))
sys.path.append(current_env)
os.system(f"pip install -r {current_env}/requirements.txt")
os.system(f"pip install {current_env}/en_core_web_sm-3.7.0-py3-none-any.whl")
from ner import tokenizer
from ner.model import BertForTaggingWithCharPooling
from ner.trainer import get_optimizer_grouped_parameters, EvaluateCallable
from ner.arguments import Arguments
from ner.metrics import Metrics
from util.read_file import Reader
from transformers import AutoConfig, BertForTokenClassification, default_data_collator, Trainer, AutoTokenizer
import config as config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("train_args")
    group.add_argument("--checkpoint_name", default="biobert")
    group.add_argument("--output_dir", default="/models")
    group.add_argument("--checkpoint", default="")
    group.add_argument("--num_train_epochs", type=int, default=50)
    group.add_argument("--datasets_index", type=int, default=0)
    group.add_argument("--use_adapter", type=int, default=0)
    group.add_argument("--use_spacy", type=int, default=0)
    group.add_argument("--use_charT", type=int, default=0)
    group.add_argument("--use_tensorboard", type=int, default=0)
    group.add_argument("--save_best", type=int, default=0)
    group.add_argument("--syntax_index", type=int, default=0)
    args = parser.parse_args()
    checkpoint_name = args.checkpoint_name
    output_dir = args.output_dir
    seed = 12
    batch_size = 64
    num_train_epochs = args.num_train_epochs
    use_datasets = config.datasets_repo[args.datasets_index]
    use_adapter = args.use_adapter != 0
    use_spacy = args.use_spacy != 0
    use_charT = args.use_charT != 0
    save_best = args.save_best != 0
    use_tensorboard = args.use_tensorboard != 0
    config_json = config.config(current_env=current_env)
    tensorboard_name = use_datasets
    spacy_tag = "spacy" if use_spacy else "no_spacy"
    tensorboard_name = checkpoint_name + "_" + tensorboard_name + "_" + spacy_tag
    checkpoint = args.checkpoint if use_charT else config_json["checkpoint"][checkpoint_name]
    tensorboard_path = f"{os.path.dirname(current_env)}/chart_logs/tensorboard_logs"
    adapter_name = tensorboard_name
    if use_adapter:
        tensorboard_name = tensorboard_name + "_" + "adapter"
    tags = config_json["tags"][use_datasets]
    datasets_path = config_json["datasets"][use_datasets]
    token_cache_index = (4 if use_spacy else 3) if use_charT else (0 if use_spacy else 1)
    args.syntax_index = 0 if not use_spacy else args.syntax_index
    token_cache_path = config.token_cache(config.token_cache_repo[token_cache_index], use_datasets, current_env, args.syntax_index)
    args.save_dir = f"{os.path.dirname(current_env)}/best_model/{use_datasets}/{config.token_cache_repo[token_cache_index]}"
    label2id = {context: i for i, context in enumerate(tags)}
    id2label = {i: context for i, context in enumerate(tags)}
    datasetDict = Reader(datasets_path).get_datasets()
    tokenizer_obj = tokenizer.NERTokenizer(checkpoint=checkpoint, label2id=label2id, id2label=id2label,
                                           datasetDict=datasetDict, token_cache_path=token_cache_path,
                                           use_spacy=use_spacy, spacy_pos_token=config.syntax_repo[args.syntax_index])
    dataloader = tokenizer_obj.get_data_with_char() if use_charT else tokenizer_obj.get_dataloader()
    train_args = Arguments.arguments(output_dir=output_dir, num_train_epochs=num_train_epochs, batch_size=batch_size,
                                     seed=seed)
    model = BertForTaggingWithCharPooling.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
        checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)) if use_charT else (
        BertForTokenClassification.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
            checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)))
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
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        data_collator=default_data_collator,
        train_dataset=dataloader["train"],
        eval_dataset=dataloader["validation"],
        compute_metrics=metrics,
        optimizers=optimizer
    )
    trainer.add_callback(EvaluateCallable(trainer=trainer, use_tensorboard=use_tensorboard, use_adapter=use_adapter,
                                          dataloader=dataloader, tensorboard_name=tensorboard_name,
                                          tensorboard_path=tensorboard_path, save_best=save_best,
                                          save_dir=args.save_dir))
    trainer.train()
