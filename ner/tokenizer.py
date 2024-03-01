import spacy
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class NERTokenizer:

    def __init__(self, checkpoint, label2id, id2label, datasetDict, token_cache_path,
                 max_sentence=128, cls="[CLS]", sep="[SEP]", use_spacy=False,
                 spacy_pos_token=None):
        if spacy_pos_token is None:
            spacy_pos_token = {"NOUN": "[unused8]", "VERB": "[unused16]"}
        self.datasetDict = datasetDict
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.token_cache_path = token_cache_path
        self.label2id = label2id
        self.id2label = id2label
        self.max_sentence = max_sentence
        self.cls = cls
        self.sep = sep
        self.use_spacy = use_spacy
        self.bert_tag_label_id = -100
        self.spacy_tag_label_id = -100
        self.pad_label_id = -100
        self.pad_token = 0
        self.spacy_pos_token = spacy_pos_token
        self.model = None
        self.split_threshold = 5
        self.max_char = 128

    def build_spacy(self, sent_list):
        spacy_model = spacy.load("en_core_web_sm")
        return {
            "syntax": spacy_model(' '.join(sent_list)),
            "find_idx": self.build_sent_idx(sent_list)
        }

    @staticmethod
    def build_sent_idx(sent_list):
        total_len = 0
        find_idx = []
        for i, word in enumerate(sent_list):
            find_idx.append(str(total_len) + "-" + str(total_len + len(word) - 1))
            # stridx2listidx[str(total_len) + "-" + str(total_len + len(word))] = i
            total_len = total_len + len(word) + 1
        return find_idx

    @staticmethod
    def do_find_idx(find_idx, idx):
        for i, func in enumerate(find_idx):
            func_list = func.split("-")
            if (int(func_list[0]) <= idx and int(func_list[-1]) >= idx):
                return i

    def check_append_word(self, tokens_list, word, charge_len=1):
        if (len(tokens_list) + 1) > self.max_sentence - charge_len:
            return
        tokens_list.append(word)

    def padding_fn(self, input_ids, token_type_ids, attention_mask, labels):
        pad_len = self.max_sentence - len(input_ids)
        input_ids += [self.pad_token] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len
        labels += [self.pad_label_id] * pad_len
        return input_ids, token_type_ids, attention_mask, labels

    def padding_char_ids(self, char_ids_list, pool_mask, pool_pad_mask):
        pad_len = self.max_sentence - len(pool_mask)
        for index, char_ids in enumerate(char_ids_list):
            pad_char_len = self.max_char - len(char_ids)
            char_ids += [0] * pad_char_len
            pad_mask = pool_pad_mask[index]
            pad_mask += [0] * pad_char_len
            char_ids_list[index] = char_ids
            pool_pad_mask[index] = pad_mask
        char_ids_list += [[0] * self.max_char] * pad_len
        pool_pad_mask += [[0] * self.max_char] * pad_len
        pool_mask += [0] * pad_len
        return char_ids_list, pool_mask, pool_pad_mask

    def tokenize_function(self, example):
        word_list = example["words"]
        original_labels = example["labels"]
        spacy_result = self.build_spacy(word_list)
        spacy = spacy_result["syntax"]
        sent_idx = spacy_result["find_idx"]
        tokens = [self.cls]
        labels = [self.bert_tag_label_id]
        last_idx = 0
        sub_token_tag = None
        for _, word in enumerate(spacy):
            original_idx = self.do_find_idx(sent_idx, word.idx)
            # the sub token from same token
            if original_idx == last_idx:
                if not sub_token_tag and self.spacy_pos_token.get(word.pos_):
                    sub_token_tag = self.spacy_pos_token.get(word.pos_)
            # the sub token from new token
            else:
                last_token = word_list[last_idx]
                first_sub_token = True
                for sub_token in self.tokenizer.tokenize(last_token):
                    self.check_append_word(tokens, sub_token)
                    label_tag = original_labels[last_idx] if first_sub_token else "O"
                    self.check_append_word(labels, self.label2id[label_tag])
                    first_sub_token = False
                # add verb or noun token
                if sub_token_tag:
                    self.check_append_word(tokens, sub_token_tag)
                    self.check_append_word(labels, self.spacy_tag_label_id)
                last_idx = original_idx
        assert len(tokens) == len(labels)
        input_ids, token_type_ids, attention_mask, labels = self.padding_fn(
            self.tokenizer.convert_tokens_to_ids(tokens),
            [0] * len(tokens),
            [1] * len(tokens),
            labels,
        )
        decoder_mask = [(x != self.pad_label_id) for x in labels]
        assert len(input_ids) == self.max_sentence
        assert len(token_type_ids) == self.max_sentence
        assert len(attention_mask) == self.max_sentence
        assert len(labels) == self.max_sentence
        assert len(decoder_mask) == self.max_sentence
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_mask": decoder_mask
        }

    def tokenize_function_no_spacy(self, example):
        tokens = []
        label_ids = []
        word_list = example["words"]
        original_labels = example["labels"]
        for word, label in zip(word_list, original_labels):
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([self.label2id[label]] + [self.pad_label_id] * (len(word_tokens) - 1))

        if len(tokens) > self.max_sentence - 2:
            tokens = tokens[:(self.max_sentence - 2)]
            label_ids = label_ids[:(self.max_sentence - 2)]

        tokens += [self.sep]
        label_ids += [self.pad_label_id]

        tokens = [self.cls] + tokens
        label_ids = [self.pad_label_id] + label_ids

        input_ids, token_type_ids, attention_mask, labels = self.padding_fn(
            self.tokenizer.convert_tokens_to_ids(tokens),
            [0] * len(tokens),
            [1] * len(tokens),
            label_ids,
        )
        decoder_mask = [(x != self.pad_label_id) for x in label_ids]
        assert len(input_ids) == self.max_sentence
        assert len(token_type_ids) == self.max_sentence
        assert len(attention_mask) == self.max_sentence
        assert len(label_ids) == self.max_sentence
        assert len(decoder_mask) == self.max_sentence
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "decoder_mask": decoder_mask
        }

    def tokenize_function_char_embeds(self, example):
        tokens = []
        label_ids = []
        word_list = example["words"]
        original_labels = example["labels"]
        char_ids_list = []
        pool_mask = []
        pool_pad_mask = []
        vocab_list = list(self.tokenizer.get_vocab())
        for word, label in zip(word_list, original_labels):
            word_tokens = self.tokenizer.tokenize(word)
            is_candidate = (not word.isalpha()) or len(word_tokens) > self.split_threshold
            pool_mask += ([1 if is_candidate else 0] * len(word_tokens))
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                char_exists = True
                for sub in word_tokens:
                    for char in sub:
                        char_exists = char in vocab_list
                        if not char_exists:
                            break
                    assert char_exists
                    sub_char_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(char))[0]
                                    for char in sub]
                    assert len(sub_char_ids) == len(sub)
                    pool_pad_mask.append([1 if is_candidate else 0] * len(sub))
                    char_ids_list.append(sub_char_ids)
                label_ids.extend([self.label2id[label]] + [self.pad_label_id] * (len(word_tokens) - 1))
        if len(tokens) > self.max_sentence - 2:
            tokens = tokens[:(self.max_sentence - 2)]
            label_ids = label_ids[:(self.max_sentence - 2)]
            pool_mask = pool_mask[:(self.max_sentence - 2)]
            char_ids_list = char_ids_list[:(self.max_sentence - 2)]
            pool_pad_mask = pool_pad_mask[:(self.max_sentence - 2)]
        tokens += [self.sep]
        label_ids += [self.pad_label_id]
        tokens = [self.cls] + tokens
        label_ids = [self.pad_label_id] + label_ids
        pool_mask = [self.pad_token] + pool_mask
        char_ids_list = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.cls))] + char_ids_list
        pool_pad_mask = [[self.pad_token]] + pool_pad_mask
        input_ids, token_type_ids, attention_mask, labels = self.padding_fn(
            self.tokenizer.convert_tokens_to_ids(tokens),
            [0] * len(tokens),
            [1] * len(tokens),
            label_ids,
        )
        char_ids_list, pool_mask, pool_pad_mask = self.padding_char_ids(char_ids_list, pool_mask, pool_pad_mask)
        decoder_mask = [(x != self.pad_label_id) for x in label_ids]
        assert len(input_ids) == self.max_sentence
        assert len(token_type_ids) == self.max_sentence
        assert len(attention_mask) == self.max_sentence
        assert len(label_ids) == self.max_sentence
        assert len(decoder_mask) == self.max_sentence
        assert len(pool_mask) == self.max_sentence
        assert len(char_ids_list) == self.max_sentence
        assert len(char_ids_list[0]) == self.max_char
        assert len(pool_pad_mask[0]) == self.max_char
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "decoder_mask": decoder_mask,
            "pool_mask": pool_mask,
            "char_ids_list": char_ids_list,
            "pool_pad_mask": pool_pad_mask
        }

    def tokenize_function_char_embeds_syntax(self, example):
        tokens = []
        label_ids = []
        word_list = example["words"]
        original_labels = example["labels"]
        char_ids_list = []
        pool_mask = []
        pool_pad_mask = []
        spacy_result = self.build_spacy(word_list)
        spacy = spacy_result["syntax"]
        sent_idx = spacy_result["find_idx"]
        last_idx = 0
        sub_token_tag = None
        vocab_list = list(self.tokenizer.get_vocab())
        charge_len = 2
        for _, word in enumerate(spacy):
            original_idx = self.do_find_idx(sent_idx, word.idx)
            original_word = word_list[original_idx]
            if original_idx == last_idx:
                if not sub_token_tag and self.spacy_pos_token.get(word.pos_):
                    sub_token_tag = self.spacy_pos_token.get(word.pos_)
            else:
                last_token = word_list[last_idx]
                first_sub_token = True
                is_candidate = ((not original_word.isalpha()) or
                                len(self.tokenizer.tokenize(original_word)) > self.split_threshold)
                for sub_token in self.tokenizer.tokenize(last_token):
                    self.check_append_word(tokens, sub_token, charge_len)
                    label_tag = original_labels[last_idx] if first_sub_token else "O"
                    self.check_append_word(label_ids, self.label2id[label_tag], charge_len)
                    first_sub_token = False
                    self.check_append_word(pool_mask, 1 if is_candidate else 0, charge_len)
                    if len(sub_token) > 0:
                        char_exists = True
                        for char in sub_token:
                            char_exists = char in vocab_list
                            if not char_exists:
                                break
                        assert char_exists
                        sub_char_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(char))[0]
                                        for char in sub_token]
                        assert len(sub_char_ids) == len(sub_token)
                        self.check_append_word(pool_pad_mask, [1 if is_candidate else 0] * len(sub_token), charge_len)
                        self.check_append_word(char_ids_list, sub_char_ids, charge_len)
                if sub_token_tag:
                    self.check_append_word(tokens, sub_token_tag, charge_len)
                    self.check_append_word(label_ids, self.spacy_tag_label_id, charge_len)
                    self.check_append_word(char_ids_list, [0], charge_len)
                    self.check_append_word(pool_pad_mask, [0], charge_len)
                    self.check_append_word(pool_mask, 0, charge_len)
                last_idx = original_idx
        assert len(tokens) == len(label_ids)
        assert len(tokens) == len(pool_mask)
        assert len(tokens) == len(pool_pad_mask)
        #sep
        tokens += [self.sep]
        label_ids += [self.pad_label_id]
        pool_mask += [self.pad_token]
        char_ids_list += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.sep))]
        pool_pad_mask += [[self.pad_token]]
        #cls
        tokens = [self.cls] + tokens
        label_ids = [self.pad_label_id] + label_ids
        pool_mask = [self.pad_token] + pool_mask
        char_ids_list = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.cls))] + char_ids_list
        pool_pad_mask = [[self.pad_token]] + pool_pad_mask
        input_ids, token_type_ids, attention_mask, labels = self.padding_fn(
            self.tokenizer.convert_tokens_to_ids(tokens),
            [0] * len(tokens),
            [1] * len(tokens),
            label_ids,
        )
        char_ids_list, pool_mask, pool_pad_mask = self.padding_char_ids(char_ids_list, pool_mask, pool_pad_mask)
        decoder_mask = [(x != self.pad_label_id) for x in label_ids]
        assert len(input_ids) == self.max_sentence
        assert len(token_type_ids) == self.max_sentence
        assert len(attention_mask) == self.max_sentence
        assert len(label_ids) == self.max_sentence
        assert len(decoder_mask) == self.max_sentence
        assert len(pool_mask) == self.max_sentence
        assert len(char_ids_list) == self.max_sentence
        assert len(char_ids_list[0]) == self.max_char
        assert len(pool_pad_mask[0]) == self.max_char
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "decoder_mask": decoder_mask,
            "pool_mask": pool_mask,
            "char_ids_list": char_ids_list,
            "pool_pad_mask": pool_pad_mask
        }

    def get_dataloader(self):
        tokenize_func = self.tokenize_function if self.use_spacy else self.tokenize_function_no_spacy
        # standard_columns = ["input_ids", "token_type_ids", "attention_mask", "labels", "decoder_mask"]
        standard_columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]
        dataloader = self.datasetDict.map(function=tokenize_func, cache_file_names=self.token_cache_path,
                                          load_from_cache_file=True)
        dataloader = dataloader.select_columns(standard_columns)
        dataloader.set_format("torch")
        return dataloader

    def get_data_with_char(self):
        standard_columns = ["input_ids", "token_type_ids", "attention_mask", "labels", "pool_mask", "char_ids_list",
                            "pool_pad_mask"]
        func = self.tokenize_function_char_embeds_syntax if self.use_spacy else self.tokenize_function_char_embeds
        dataloader = self.datasetDict.map(function=func,
                                          cache_file_names=self.token_cache_path,
                                          load_from_cache_file=True)
        dataloader = dataloader.select_columns(standard_columns)
        dataloader.set_format("torch")
        return dataloader

class CHARTokenizer:
    def __init__(self, checkpoint, datasetDict, max_sentence=128, cls="[CLS]", sep="[SEP]"):
        self.datasetDict = datasetDict
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=False)
        self.cls = cls
        self.sep = sep
        self.pad_token = 0
        self.special_token_map = {
            " ": "[unused10]"
        }
        self.max_sentence = max_sentence

    def padding_fn(self, input_ids, attention_mask):
        pad_len = self.max_sentence - len(input_ids)
        if pad_len > 0:
            input_ids += [self.pad_token] * pad_len
            attention_mask += [0] * pad_len
        elif pad_len < 0:
            input_ids = input_ids[:self.max_sentence]
            attention_mask = attention_mask[:self.max_sentence]
        labels = input_ids.copy()
        return input_ids, attention_mask, labels

    def tokenize_function(self, example):
        word = example["words"]
        word_tokens = [self.cls]
        for char in word:
            char = self.special_token_map[char] if (char in self.special_token_map.keys()) else char
            word_tokens = word_tokens + [char]
        word_tokens = word_tokens + [self.sep]
        input_ids, attention_mask, labels = self.padding_fn(input_ids=self.tokenizer.convert_tokens_to_ids(word_tokens),
                                                            attention_mask=[1] * len(word_tokens))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def datasets_char_map(self, mlm_probability):
        standard_columns = ["input_ids", "attention_mask", "labels"]
        datasets = self.datasetDict.map(function=self.tokenize_function)
        datasets = datasets.select_columns(standard_columns)
        datasets.set_format("torch")
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_probability)
        return datasets, data_collator

    def reform_special_charater(self, word):
        special_charater_prefix = ["##"]
        special_charater_post = []
        for char in special_charater_prefix:
            if word.startswith(char):
                word = word[len(char):]
        for char in special_charater_post:
            if word.endswith(char):
                word = word[:len(char)]
        return word

    def put_model_char_embedding(self, model):
        handle_words = []
        total_word_list = [single for words in self.datasetDict["train"]["words_list"] for single in words]
        for word in total_word_list:
            if word in handle_words:
                continue
            handle_words.append(word)
            sub_words_list = self.tokenizer.tokenize(word)
            for sub_word in sub_words_list:
                sub_word_chars = []
                sub_word_id = self.tokenizer.convert_tokens_to_ids(sub_word)
                # sub_word = self.reform_special_charater(sub_word)
                for char in sub_word:
                    sub_word_chars.append(char)
                sub_word_char_ids = self.tokenizer.convert_tokens_to_ids(sub_word_chars)
                embed = model.bert.embeddings.word_embeddings.weight.data[sub_word_char_ids[0]]
                for i in sub_word_char_ids[1:]:
                    embed += model.bert.embeddings.word_embeddings.weight.data[i]
                embed /= len(sub_word_char_ids)
                model.bert.embeddings.word_embeddings.weight.data[sub_word_id] += embed
                model.bert.embeddings.word_embeddings.weight.data[sub_word_id] /= 2
