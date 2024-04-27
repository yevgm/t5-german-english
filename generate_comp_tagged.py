import argparse
from config import parse_global_args
from src.utils.processing import *
from src.dataset import load_datasets
from src.trainer import get_trainer_and_tokenizer
import os

def main():
    """
    Generates comp competition tagged file by using competition model3
    """

    # parse from arguments command line
    empty_parser = argparse.ArgumentParser()
    parser = parse_global_args(parent=empty_parser)
    run_config = parser.parse_args()  # parse args from cli
    set_seed(run_config.seed)

    # load model
    run_config.train = False

    trainer, tokenizer = get_trainer_and_tokenizer(run_config)

    trainer.load_model(run_config)

    id1 = 987654321
    id2 = 123456789

    # Competition
    print('Start predicting comp\n')
    dataloaders = load_datasets(run_config=run_config, tokenizer=tokenizer, huggingface=False, test_type="comp")
    _, pred_en_list = trainer.predict(test_datalaoder=dataloaders[2])
    print('Done predicting comp\n')

    comp_out = 'comp_' + str(id1) + '_' + str(id2) + '.labeled'
    de_sen_list, _, _ = read_file_unlabeled_new_line(os.path.join(run_config.repo_root, 'data', 'comp.unlabeled'))
    write_preds(de_sen_list, pred_en_list, comp_out)

    # Validation
    print('Start predicting val\n')
    dataloaders = load_datasets(run_config=run_config, tokenizer=tokenizer, huggingface=False, test_type="val")
    _, pred_en_list = trainer.predict(test_datalaoder=dataloaders[2])
    print('Done predicting val\n')

    val_out = 'val_' + str(id1) + '_' + str(id2) + '.labeled'
    de_sen_list, _, _ = read_file_unlabeled_new_line(os.path.join(run_config.repo_root, 'data', 'val.unlabeled'))
    write_preds(de_sen_list, pred_en_list, val_out)


if __name__ == '__main__':
    main()
