import sys
sys.path.append("/ai/zyf/project")
from ner import tokenizer
from ner.model import BertForTagging, AdapterForTagging
from ner.trainer import Trainer, get_optimizer_grouped_parameters, EvaluateCallable
from ner.arguments import Arguments
import torch
from ner.metrics import Metrics
from util.read_file import Reader
from transformers import AutoConfig
import json

if __name__ == "__main__":
    checkpoint_name = "biobert"
    out_putdir = "/models"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 12
    batch_size = 32
    num_train_epochs = 50
    datasets_repo = ["chemdner", "cdr", "chemrxnextractor"]
    use_datasets = datasets_repo[0]
    use_adapter = True
    use_spacy = False
    use_tensorboard = True
    config = json.load(open('/ai/zyf/project/config.py'))
    tensorboard_name = use_datasets
    spacy_tag = "syntax" if use_spacy else "no_spacy"
    tensorboard_name = checkpoint_name + "_" + tensorboard_name + "_" + spacy_tag
    checkpoint = config["checkpoint"][checkpoint_name]
    adapter_name = tensorboard_name
    if use_adapter:
        tensorboard_name = tensorboard_name + "_" + "adapter"
    tags = config["tags"][use_datasets]
    datasets_path = config["datasets"][use_datasets]
    token_cache_path = config["token_cache"][use_datasets][spacy_tag]
    label2id = {context: i for i, context in enumerate(tags)}
    id2label = {i: context for i, context in enumerate(tags)}
    datasetDict = Reader(datasets_path).get_datasets()
    tokenizer_obj = tokenizer.NERTokenizer(checkpoint=checkpoint, label2id=label2id, id2label=id2label,
                                           datasetDict=datasetDict, token_cache_path=token_cache_path,
                                           use_spacy=use_spacy)
    dataloader = tokenizer_obj.get_dataloader()
    train_args = Arguments.arguments(output_dir=out_putdir, num_train_epochs=num_train_epochs, batch_size=batch_size,
                                     seed=seed)

    model = AdapterForTagging.from_pretrained(checkpoint=checkpoint, adapter_name=adapter_name, num_labels=len(tags)) \
        if use_adapter else BertForTagging.from_pretrained(
        checkpoint, config=AutoConfig.from_pretrained(checkpoint, num_labels=len(id2label), id2label=id2label,
                                                      label2id=label2id))
    metrics_obj = Metrics(id2label=id2label)
    dataset_len = len(dataloader["train"])
    total_steps = (dataset_len // batch_size) * train_args.num_train_epochs if dataset_len % batch_size == 0 else \
        (dataset_len // batch_size + 1) * train_args.num_train_epochs
    train_args.warmup_steps = 0.1 * total_steps
    metrics = metrics_obj.adapter_metrics if use_adapter else metrics_obj.ner_metrics
    optimizer = None if use_adapter else get_optimizer_grouped_parameters(args=train_args, model=model,
                                                                          num_training_steps=total_steps)
    trainer = Trainer.trainer(use_adapter=use_adapter, train_args=train_args, model=model,
                              tokenizer=tokenizer_obj.tokenizer, dataloader=dataloader, metrics=metrics,
                              optimizer=optimizer, device=device)
    trainer.add_callback(EvaluateCallable(trainer=trainer, use_tensorboard=use_tensorboard, use_adapter=use_adapter,
                                          dataloader=dataloader, tensorboard_name=tensorboard_name))
    trainer.train()
