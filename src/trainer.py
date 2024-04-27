from transformers import Trainer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from typing import List, Optional
import os
from transformers import get_scheduler
from src.dataset import HuggingFaceCollate
from transformers import TrainingArguments
from src.dataset import load_datasets
from src.utils.processing import get_model


class CustomTrainer(Trainer):
    def predict(self, test_datalaoder: DataLoader, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"):
        self.model.eval()
        pred_en_list = []
        kwargs = {'max_length': 300, 'num_beams': 8, 'early_stopping': False, 'min_length': 5}
        tokenizer = test_datalaoder.dataset.tokenizer
        with torch.no_grad():
            test_de_list = []
            for batch in tqdm(test_datalaoder, total=len(test_datalaoder)):
                input_ids, attention_mask, labels, de_sen, en_sen = batch

                outputs = self.model.generate(input_ids, **kwargs)
                pred_en_sen = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # post-processing. Fix bug of empty prediction
                for i, sen in enumerate(pred_en_sen):
                    if len(sen) == 0 or sen == '\n' or sen == '\t':
                        pred_en_sen[i] = de_sen[i]

                pred_en_list += pred_en_sen
                test_de_list += de_sen

        return test_de_list, pred_en_list

    def save_last_model(self, run_config):
        os.makedirs(run_config.output_dir, exist_ok=True)
        model_fp = os.path.join(run_config.output_dir, run_config.run_name) + '.ckp'
        saved_dict = {'model_checkpoint': self.model.state_dict()}
        torch.save(saved_dict, model_fp)

    def load_model(self, run_config):
        weights_model = f"{run_config.output_dir}/{run_config.load}"
        checkpoint = torch.load(weights_model, map_location=run_config.device)
        self.model.load_state_dict(checkpoint['model_checkpoint'])
        self.model.to(run_config.device)


def get_trainer_and_tokenizer(run_config):
    model, tokenizer = get_model(run_config)
    model.to(run_config.device)
    dataloaders = load_datasets(run_config=run_config, tokenizer=tokenizer, huggingface=True)

    num_training_steps = run_config.epochs * len(dataloaders[0].dataset) // (
            run_config.batch_size * run_config.grad_accu_steps)
    args = TrainingArguments(output_dir='./output',
                             per_device_train_batch_size=run_config.batch_size,
                             save_strategy='no',
                             logging_steps=num_training_steps // (2 * run_config.epochs),  # every 1/2 epoch
                             evaluation_strategy='no',
                             do_train=run_config.train,
                             dataloader_pin_memory=False,
                             num_train_epochs=run_config.epochs,
                             report_to=['none'],
                             fp16=run_config.fp16,
                             gradient_accumulation_steps=run_config.grad_accu_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)

    print('LR num of training steps: {}'.format(num_training_steps))
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    my_collate = HuggingFaceCollate(tokenizer, run_config, huggingface=True)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=dataloaders[0].dataset,
        optimizers=(optimizer, scheduler),
        data_collator=my_collate
    )

    return trainer, tokenizer
