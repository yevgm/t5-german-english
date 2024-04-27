from torch.utils.data import Dataset, DataLoader
import os
from src.utils.processing import read_file_labeled, read_file_unlabeled
import numpy as np


class HuggingFaceCollate:
    def __init__(self, tokenizer, run_config, huggingface=False, test_type="val"):
        self.tokenizer = tokenizer
        self.run_config = run_config
        self.device = run_config.device
        self.huggingface = huggingface
        self.test_type = test_type

    def __call__(self, batch):
        max_source_length = self.run_config.source_length
        max_target_length = 512

        task_prefix = "translate German to English: "

        # encode the inputs
        if self.test_type == "comp":
            de_sen = batch
        elif self.test_type == "val":
            de_sen = [tuple_[1] for tuple_ in batch]
        else:
            raise NotImplementedError(f"{self.test_type} is not supported!")

        encoding = self.tokenizer(
            [task_prefix + sen for sen in de_sen],
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt")

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # encode the targets
        if self.test_type == "val":
            en_sen = [tuple_[0] for tuple_ in batch]
            target_encoding = self.tokenizer(
                en_sen,
                padding="longest",
                max_length=max_target_length,
                truncation=True,
                return_tensors="pt")

            labels = target_encoding.input_ids

            # replace padding token id's of the labels by -100, so it's ignored by the loss
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels = labels.to(self.device)

        if self.huggingface:
            if self.test_type == "comp":
                output = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': None}
            elif self.test_type == "val":
                output = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
            else:
                raise NotImplementedError(f"{self.test_type} is not supported!")
            return output
        else:
            if self.test_type == "comp":
                return input_ids, attention_mask, None, de_sen, None
            elif self.test_type == "val":
                return input_ids, attention_mask, labels, de_sen, en_sen
            else:
                raise NotImplementedError(f"{self.test_type} is not supported!")


class TranslationDataset(Dataset):
    """
    Pytorch dataset class.
    """

    def __init__(self, run_config, raw_data, tokenizer, comp=False):
        """
        Dataset constructor.
        """
        self.run_config = run_config
        if comp:
            self.size = len(raw_data)  # number of sentences in data
            self.de_sen_list = raw_data
        else:
            self.size = len(raw_data[0])  # number of sentences in data
            self.en_sen_list, self.de_sen_list = raw_data[0], raw_data[1]
        self.tokenizer = tokenizer
        self.comp = comp

    def __getitem__(self, idx: int):
        """
        Get the sentence specified by idx, and load to device
        """
        if self.comp:
            return self.de_sen_list[idx]
        else:
            return self.en_sen_list[idx], self.de_sen_list[idx]

    def __len__(self):
        return self.size


def filter_data(dataset):
    new_train_data = [[], []]
    bad_sentences = [[], []]
    for de_sen, en_sen in zip(dataset[0], dataset[1]):
        n = 10
        if len(de_sen) >= n and len(en_sen) >= n:
            new_train_data[0].append(de_sen)
            new_train_data[1].append(en_sen)
        else:
            bad_sentences[0].append(de_sen)
            bad_sentences[1].append(en_sen)

    return new_train_data


def load_datasets(run_config, tokenizer, huggingface=False, test_type="val"):
    """
    Return DataLoaders of Train and Test
    """
    train_fp = os.path.join(run_config.repo_root, 'data', 'train.labeled')
    train_data = read_file_labeled(train_fp)

    if test_type == "comp":
        comp_file_path = os.path.join(run_config.repo_root, 'data', 'comp.unlabeled')
        de_sen_list, en_roots, en_mods = read_file_unlabeled(comp_file_path)
    elif test_type == "val":
        test_fp = os.path.join(run_config.repo_root, 'data', 'val.labeled')
        test_data = read_file_labeled(test_fp)
    else:
        raise NotImplementedError(f"{test_type} is not supported!")

    # split train to train data and validation data
    if not run_config.no_validation_set:
        n_train = len(train_data[0])
        val_ind = np.random.choice(range(n_train), size=(n_train // 5,), replace=False).astype(int)
        set_intersection = set(range(n_train)).intersection(set(val_ind))
        train_ind = set(range(n_train)) - set_intersection
        train_data_, val_data_ = [], []
        for j in range(2):
            train_data_.append([train_data[j][i] for i in train_ind])
            val_data_.append([train_data[j][i] for i in val_ind])

        train_data = tuple(train_data_)
        validation_data = tuple(val_data_)
    else:
        validation_data = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ for debugging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if run_config.debug and test_type == "val":
        train_data = (train_data[0][:100], train_data[1][:100])
        test_data = (test_data[0][:5], test_data[1][:5])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    train_dataset = TranslationDataset(run_config=run_config, raw_data=train_data, tokenizer=tokenizer)
    if test_type == "comp":
        test_dataset = TranslationDataset(run_config=run_config, raw_data=de_sen_list, tokenizer=tokenizer, comp=True)
    elif test_type == "val":
        test_dataset = TranslationDataset(run_config=run_config, raw_data=test_data, tokenizer=tokenizer)
    else:
        raise NotImplementedError(f"{test_type} is not supported!")

    my_collate = HuggingFaceCollate(tokenizer, run_config, huggingface=huggingface, test_type=test_type)

    # defining the dataloader class for each dataset
    train_loader = DataLoader(train_dataset, batch_size=run_config.batch_size, shuffle=True,
                              collate_fn=my_collate)

    if not run_config.no_validation_set:
        validation_dataset = TranslationDataset(run_config=run_config, raw_data=validation_data, tokenizer=tokenizer)
        val_loader = DataLoader(validation_dataset, batch_size=run_config.batch_size, shuffle=False,
                                collate_fn=my_collate)
    else:
        val_loader = []

    test_loader = DataLoader(test_dataset, batch_size=run_config.inference_batch_size, shuffle=False,
                             collate_fn=my_collate)

    return train_loader, val_loader, test_loader
