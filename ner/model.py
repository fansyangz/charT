from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification, AutoModel
from torch.nn import CrossEntropyLoss
try:
    from transformers.adapters import AutoAdapterModel
except ImportError:
    from transformers import AutoModel as AutoAdapterModel

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


class BertForTagging(BertForTokenClassification):
    def __init__(self, config):
        super(BertForTagging, self).__init__(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, decoder_mask=None, ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        loss_fct = CrossEntropyLoss()
        # loss_fct = FocalLoss(alpha=torch.tensor([0.1, 0.45, 0.45], device=device))
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
    def decode(self, logits, mask):
        preds = torch.argmax(logits, dim=2).cpu().numpy()
        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    preds_list[i].append(preds[i, j])
        return preds_list

    def decode_labels(self, labels, mask):
        labels_np = labels.cpu().numpy()
        batch_size, seq_len = labels_np.shape
        labels_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    labels_list[i].append(labels_np[i, j])
        # labels_list_test = [labels[i, mask[i]].tolist() for i in range(labels.shape[0])]
        return labels_list


class BertForTaggingWithCharPooling(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.pooler = Pooler(dim=config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, decoder_mask=None,
                pool_mask=None, char_ids_list=None, pool_pad_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        # char_ids_embeds: b, sentence_l, char_l, dim
        char_ids_embeds = self.bert.embeddings.word_embeddings(char_ids_list)
        pool_h = self.pooler.pooling(char_ids_embeds, pool_pad_mask)
        batch_size, seq_length, hidden_dim = sequence_output.shape
        extended_h = pool_h.expand(batch_size, seq_length, hidden_dim)
        extended_sequence = torch.cat([sequence_output, extended_h], 2)
        logits = self.classifier(extended_sequence)
        outputs = (logits,) + outputs[2:]
        loss_fct = CrossEntropyLoss()
        # loss_fct = FocalLoss(alpha=torch.tensor([0.1, 0.45, 0.45], device=device))
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss,) + outputs

        return outputs



class AdapterForTagging:
    def __init__(self):
        raise EnvironmentError("can not to be instantiated")

    @classmethod
    def from_pretrained(cls, checkpoint, adapter_name, num_labels):
        model = AutoAdapterModel.from_pretrained(checkpoint)
        model.add_adapter(adapter_name)
        model.train_adapter(adapter_name)
        model.add_tagging_head(adapter_name, num_labels=num_labels)
        model.set_active_adapters(adapter_name)
        return model


class Pooler(nn.Module):
    def __init__(self, dim):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def pooling(self, char_ids_embeds, mask):
        batch_size, sentcen_len, char_len, dim = char_ids_embeds.shape
        char_ids_embeds = char_ids_embeds.reshape(-1, char_len, dim)
        extended_mask = mask.reshape(-1, char_len).unsqueeze(1)
        h = torch.bmm(extended_mask.float(), char_ids_embeds).squeeze(1)
        h = self.activation(self.dense(h))
        h = h.reshape(batch_size, sentcen_len, dim)
        return h # batch_size, sentence_l, dim