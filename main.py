import argparse
from config import parse_global_args
from src.utils.processing import *
import os
from src.trainer import get_trainer_and_tokenizer
from src.dataset import load_datasets


def main():
    # parse from arguments command line
    empty_parser = argparse.ArgumentParser()
    parser = parse_global_args(parent=empty_parser)
    run_config = parser.parse_args()  # parse args from cli
    set_seed(run_config.seed)

    # set maximum cpu threads
    torch.set_num_threads(run_config.max_cpu)

    trainer, tokenizer = get_trainer_and_tokenizer(run_config)

    if run_config.train:
        print('Start training\n')
        # Train model
        trainer.train()
        print('Done training!\n')

        # save trained model
        trainer.save_last_model(run_config)
    else:
        trainer.load_model(run_config)

    print('Start predicting\n')
    dataloaders = load_datasets(run_config=run_config, tokenizer=tokenizer, huggingface=False)
    _, pred_en_list = trainer.predict(test_datalaoder=dataloaders[2])

    print('Done predicting!\n')

    # Write prediction file and calculate sacre-bleu
    id1 = 987654321
    id2 = 123456789

    _, de_sen_list = read_file_new_line(os.path.join(run_config.repo_root, 'data', 'val.labeled'))

    val_out = 'val_' + str(id1) + '_' + str(id2) + '.labeled'
    write_preds(de_sen_list, pred_en_list, val_out)

    # Calculate BLEU
    true_fp = os.path.join(run_config.repo_root, 'data', 'val.labeled')
    tagged_fp = os.path.join(run_config.repo_root, val_out)
    calculate_score(tagged_fp, true_fp)


if __name__ == "__main__":
    main()
