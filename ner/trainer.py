from transformers import TrainingArguments, Trainer, PreTrainedModel, PreTrainedTokenizerBase, default_data_collator
try:
    from transformers.adapters import AdapterTrainer
except ImportError:
    from transformers import Trainer as AdapterTrainer
from typing import Optional, Tuple, Dict
from datasets import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.data.data_collator import DataCollator
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers.trainer_callback import DefaultFlowCallback, TrainerState, TrainerControl
import shutil
from pathlib import Path

class NERTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel,
            args: TrainingArguments,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            compute_metrics=None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
            device=torch.device("cpu")
    ):
        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
        )
        self.device = device

    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        return self._prediction_loop(self.get_eval_dataloader(eval_dataset))

    def predict(self, test_dataset: Dataset):
        return self._prediction_loop(self.get_test_dataloader(test_dataset))

    def _prediction_loop(self, dataloader: DataLoader):
        model = self.model

        model.eval()

        eval_losses = 0
        preds_ids = []
        label_ids = []
        input_ids_list = []
        t = tqdm(dataloader)
        for dataset in t:
            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            token_type_ids = dataset['token_type_ids']
            labels = dataset['labels']
            decoder_mask = dataset['decoder_mask']

            input_ids = input_ids.to(self.device, dtype=torch.long)
            attention_mask = attention_mask.to(self.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
            labels = labels.to(self.device, dtype=torch.long)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    decoder_mask=decoder_mask,
                )

            loss = outputs[0]
            eval_losses += loss.item()
            decode_predictions = model.decode(outputs[1], decoder_mask)
            decode_labels = [labels[i, decoder_mask.to(torch.bool)[i]].tolist() for i in range(labels.shape[0])]
            preds_ids.extend(decode_predictions)
            label_ids.extend(decode_labels)
            input_ids_list.extend(input_ids)
            assert len(decode_predictions) == len(decode_labels)
            batch_size, _ = labels.shape
            for i in range(batch_size):
                assert len(decode_predictions[i]) == len(decode_labels[i])
        t.close()
        metrics = self.compute_metrics(preds_ids, label_ids)
        metrics['loss'] = eval_losses / len(t)
        return {"predictions": preds_ids, "label_ids": label_ids, "metrics": metrics, "input_ids_list": input_ids_list}


class Trainer:
    def __init__(self):
        raise EnvironmentError("can not to be instantiated")

    @classmethod
    def trainer(cls, use_adapter, train_args, model, tokenizer,  dataloader, metrics, collator=default_data_collator,
                optimizer=None, device=torch.device("cpu")):
        if use_adapter:
            return AdapterTrainer(
                    model=model,
                    args=train_args,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=dataloader["train"],
                    eval_dataset=dataloader["validation"],
                    compute_metrics=metrics,
            )
        else:
            return NERTrainer(
                    model=model,
                    args=train_args,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=dataloader["train"],
                    eval_dataset=dataloader["validation"],
                    compute_metrics=metrics,
                    optimizers=optimizer,
                    device=device
                )


class EvaluateCallable(DefaultFlowCallback):
    def __init__(self, trainer, use_tensorboard, dataloader, use_adapter, tensorboard_path,
                 tensorboard_name="unknown", save_best=False, save_dir=None):
        super().__init__()
        self.trainer = trainer
        self.use_tensorboard = use_tensorboard
        self.dataloader = dataloader
        self.use_adapter = use_adapter
        self.tensorboard_name = tensorboard_name
        self.tensorboard_path = tensorboard_path
        self.save_best = save_best
        self.save_dir = save_dir
        self.first = True
        self.best_val = 0.0

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        evaluate = self.trainer.evaluate()
        print("-------------epoch: " + str(state.epoch) + "-------------")
        print("-------------evaluate: " + str(state.epoch) + "-------------")
        # if self.use_adapter:
        #     print(evaluate)
        # else:
        #     print(evaluate["metrics"])
        print(evaluate)
        test = self.trainer.predict(self.dataloader["test"])
        print("-------------test: " + str(state.epoch) + "-------------")
        # if self.use_adapter:
        #     print(test)
        # else:
        #     print(test["metrics"])
        print(test.metrics)
        test_f1 = test.metrics["test_f1"]
        if test_f1 > self.best_val:
            print(f"this epcoh is best ! the test f1: {test_f1}")
            if self.save_best and self.save_dir:
                if self.first:
                    self.first = False
                else:
                    del_file(self.save_dir)
                self.trainer.save_model(self.save_dir)
                print(f"best model save success !")
            self.best_val = test_f1
        else:
            print(f"this epoch is not best ! current best is {self.best_val}, this epoch is {test_f1}")
        if self.use_tensorboard:
            writer = SummaryWriter(self.tensorboard_path)
            writer.add_scalars(self.tensorboard_name, {"test_f1": test.metrics["test_f1"],
                                                       "eval_f1": evaluate["eval_f1"]}, state.epoch)
                # wandb.log({
                #     "eval_precision": evaluate["eval_precision"],
                #     "eval_recall": evaluate["eval_recall"],
                #     "eval_f1": evaluate["eval_f1"],
                #     # "eval_loss": evaluate["loss"],
                #     "test_precision": test.metrics["test_precision"],
                #     "test_recall": test.metrics["test_recall"],
                #     "test_f1": test.metrics["test_f1"],
                #     # "test_loss": test["loss"],
                # })
            # else:
            #     writer.add_scalars(self.tensorboard_name, {"test_f1": test["metrics"]["f1"],
            #                                                "eval_f1": evaluate["metrics"]["f1"]}, state.epoch)
                # wandb.log({
                #     "eval_precision": evaluate["metrics"]["precision"],
                #     "eval_recall": evaluate["metrics"]["recall"],
                #     "eval_f1": evaluate["metrics"]["f1"],
                #     "eval_loss": evaluate["metrics"]["loss"],
                #     "test_precision": test["metrics"]["precision"],
                #     "test_recall": test["metrics"]["recall"],
                #     "test_f1": test["metrics"]["f1"],
                #     "test_loss": test["metrics"]["loss"],
                # })


def get_optimizer_grouped_parameters(args, model, num_training_steps):
    no_decay = ['bias', 'gamma', 'beta']
    args.weight_decay = 0.01
    args.learning_rate = 3e-5
    args.adam_epsilon = 1e-8
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler

def  del_file(path):
      for elm in Path(path).glob('*'):
            print(elm)
            elm.unlink() if elm.is_file() else shutil.rmtree(elm)
