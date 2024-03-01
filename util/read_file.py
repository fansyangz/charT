from datasets import Dataset, DatasetDict


class Reader:
    def __init__(self, file_path_json, tags_pre_start="B", tags_pre_mid="I", tags_pre_other="O"):
        self.train = file_path_json["train"]
        self.dev = file_path_json["dev"]
        self.test = file_path_json["test"]
        self.tags_pre_start = tags_pre_start
        self.tags_pre_mid = tags_pre_mid
        self.tags_pre_other = tags_pre_other

    def read_examples_from_file(self, file_path):
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words, labels = [], []
            for line in f:
                line = line.rstrip()
                if line.startswith("#\tpassage"):
                    continue
                elif line == "":
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

    def get_datasets(self, just_test=False, handle_train=None):
        train_input = self.read_examples_from_file(self.train)
        dev_input = self.read_examples_from_file(self.dev)
        test_input = self.read_examples_from_file(self.test)
        train_input = train_input + dev_input if just_test else train_input
        if handle_train:
            train_input = handle_train(train_input)
        train_dataset = Dataset.from_list(train_input)
        dev_dataset = Dataset.from_list(dev_input)
        test_dataset = Dataset.from_list(test_input)
        return DatasetDict({
                    "train": train_dataset,
                    "validation": dev_dataset,
                    "test": test_dataset
                })

    def charge_tags(self, tags_charge_list, tag):
        for tags_charge in tags_charge_list:
            if tag.startswith(tags_charge):
                return True
        return False

    def get_label_words(self):
        datasetDict = self.get_datasets(just_test=True)
        data_keys = list(datasetDict.keys())
        datasetDict_map = {}
        for key in data_keys:
            label_words = []
            for sentence, labels in zip(datasetDict[key]["words"], datasetDict[key]["labels"]):
                words = []
                for index, word in enumerate(sentence):
                    label = labels[index]
                    is_last = (index + 1) == len(sentence)
                    if self.charge_tags([self.tags_pre_start], label):
                        words.append(word)
                    elif self.charge_tags([self.tags_pre_mid], label):
                        if is_last or self.charge_tags([self.tags_pre_other], labels[index + 1]):
                            words.append(word)
                            label_words.append({"words": " ".join(words), "words_list": words.copy()})
                            words = []
                        else:
                            words.append(word)
            datasetDict_map[key] = Dataset.from_list(label_words)
        return DatasetDict(datasetDict_map)

