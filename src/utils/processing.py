import evaluate
import torch
import numpy as np
import random
from typing import List, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer


def read_file_labeled(file_path: str) -> Tuple[List[str], List[str]]:
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def extract_modifiers(modifiers_str: str) -> List[Tuple[str, str]]:
    i = 0
    modifiers = []
    while i < len(modifiers_str):
        if modifiers_str[i] == '(':
            i += 1
            first_word = ''
            while not modifiers_str[i] in [',', ')']:
                first_word += modifiers_str[i]
                i += 1

            if modifiers_str[i] == ',':
                i += 1

            second_word = ''
            while modifiers_str[i] != ')':
                second_word += modifiers_str[i]
                i += 1
            i += 1  # Reached to ')', so need to update i by 1

            first_word = first_word.strip()
            second_word = second_word.strip()
            if first_word != '' and second_word != '':
                modifiers.append((first_word, second_word))
        else:
            i += 1
    return modifiers


def read_file_unlabeled(file_path: str) -> Tuple[List[str], List[List[str]], List[List[Tuple[str, str]]]]:
    de_sens, en_roots, en_mods = [], [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str = ''
        for line in f.readlines():
            line = line.strip()
            if 'Roots in English:' in line:
                roots = line.split('Roots in English: ')[-1].split(',')
                en_roots.append([root.strip() for root in roots])
                continue
            if 'Modifiers in English:' in line:
                modifiers = line.split('Modifiers in English: ')[-1]
                modifiers = extract_modifiers(modifiers)
                en_mods.append(modifiers)
                continue
            if line == 'German:':
                if len(cur_str) > 0:
                    de_sens.append(cur_str)
                    cur_str = ''
                continue
            if line == '':
                continue
            cur_str += line
    if len(cur_str) > 0:
        de_sens.append(cur_str)
    return de_sens, en_roots, en_mods


def write_preds(de_sen_list: List[str], en_sen_list: List[str], file: str) -> None:
    with open(file, "w", encoding='utf-8') as output_file:
        for de_sen, en_sen in zip(de_sen_list, en_sen_list):
            output_file.write("German:\n")
            output_file.write(de_sen)
            output_file.write("English:\n")
            output_file.write(en_sen + "\n\n")


def set_seed(seed: int) -> None:
    """
    Set global seed for the run
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(config):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    return model, tokenizer


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(tagged_en, true_en):
    metric = evaluate.load("sacrebleu")
    tagged_en = [x.strip().lower() for x in tagged_en]
    true_en = [x.strip().lower() for x in true_en]

    result = metric.compute(predictions=tagged_en, references=true_en)
    result = result['score']
    result = round(result, 2)
    return result


def read_file(file_path):
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line + ' '
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def calculate_score(file_path1, file_path2):
    file1_en, file1_de = read_file(file_path1)
    file2_en, file2_de = read_file(file_path2)
    for sen1, sen2 in zip(file1_de, file2_de):
        if sen1.strip().lower() != sen2.strip().lower():
            raise ValueError('Different Sentences')
    score = compute_metrics(file1_en, file2_en)
    print(score)


def read_file_new_line(file_path: str) -> Tuple[List[str], List[str]]:
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line + '\n'
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def read_file_unlabeled_new_line(file_path: str) -> Tuple[List[str], List[List[str]], List[List[Tuple[str, str]]]]:
    de_sens, en_roots, en_mods = [], [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str = ''
        for line in f.readlines():
            line = line.strip()
            if 'Roots in English:' in line:
                roots = line.split('Roots in English: ')[-1].split(',')
                en_roots.append([root.strip() for root in roots])
                continue
            if 'Modifiers in English:' in line:
                modifiers = line.split('Modifiers in English: ')[-1]
                modifiers = extract_modifiers(modifiers)
                en_mods.append(modifiers)
                continue
            if line == 'German:':
                if len(cur_str) > 0:
                    de_sens.append(cur_str)
                    cur_str = ''
                continue
            if line == '':
                continue
            cur_str += line + '\n'
    if len(cur_str) > 0:
        de_sens.append(cur_str)
    return de_sens, en_roots, en_mods
