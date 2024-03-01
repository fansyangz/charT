from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import EvalPrediction
import numpy as np


class Metrics:
    def __init__(self, id2label):
        self.id2label = id2label

    def ner_metrics(self, predictions, label_ids):
        label_list = [[self.id2label[x] for x in seq] for seq in label_ids]
        preds_list = [[self.id2label[x] for x in seq] for seq in predictions]

        return {
            "precision": precision_score(label_list, preds_list),
            "recall": recall_score(label_list, preds_list),
            "f1": f1_score(label_list, preds_list),
        }

    def metrics_for_trainer(self, p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        label_ids = [[l for l in label if l != -100] for label in labels]
        predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                       zip(preds, labels)]
        return self.ner_metrics(predictions=predictions, label_ids=label_ids)
