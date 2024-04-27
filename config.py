import argparse
import os
import torch


# This function is used to correctly parse boolean values from command line
def str2bool(v):
    """transfer str to bool for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_global_args(parent, add_help=False):
    """
        Parse commandline arguments.
        """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    # general
    repo_root = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(repo_root, 'output')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument('--repo_root', default=repo_root, type=str, help='repository main dir')
    parser.add_argument('--output_dir', default=output_folder, type=str, help='output dir')
    parser.add_argument('--save_model', default=True, type=str2bool,
                        help='save model to output folder or not')
    parser.add_argument('--run_name', default='last_model', type=str, help='placeholder for wandb run name')
    parser.add_argument('--seed', default=42, type=int, help='random seed for everything')
    parser.add_argument('--device', default=device, type=str, help='which device to train')
    parser.add_argument('--train', default=True, type=str2bool, help='to train or to evaluate')
    parser.add_argument('--load', default='last_model.ckp', type=str, help='Which model to load')
    parser.add_argument('--max_cpu', type=int, default=500, help='maximum number of cpu threads')

    # learning
    parser.add_argument('--source_length', type=int, default=220, help='maximum length of input tokens')
    parser.add_argument('--epochs', type=int, default=40, help='Total number of epochs')
    parser.add_argument('--model', type=str, default='t5', help='Model name')
    parser.add_argument('--val_step_every', type=int, default=1, help='run validation set every x number of epochs')
    parser.add_argument('--fp16', default=True, type=str2bool, help='train with 16 bit precision forward pass')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Regularization term')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--grad_accu_steps', type=int, default=4, help='training accumulate gradient size')
    parser.add_argument('--inference_batch_size', type=int, default=2, help='inference batch size')
    parser.add_argument('--no_validation_set', default=True, type=str2bool,
                        help='use full test set, or split to validation')

    # debug
    parser.add_argument('--debug', default=False, type=str2bool, help='Debug flag to restrict datasets\' size')

    return parser
