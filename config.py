import argparse


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=2, help='Experiment number of the trial run.')
    parse.add_argument('--data_dir', type=str, default='../autodl-fs')
    parse.add_argument('--lr', type=float, default=1e-6, help='learning rate for noise predictor')
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--seed', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=10)
    parse.add_argument('--log_dir', default='../tf-logs')
    parse.add_argument('--n_class', default=21)
    parse.add_argument('--use_amp', default=False)
    parse.add_argument('--grad_accum', default=1)
    parse.add_argument('--n_samples', default=1, help='generate augmentation scale')
    parse.add_argument('--log_file', default='./log.txt')
    parse.add_argument('--dataset', default='endnet')     # szutree2, endnet
    args = parse.parse_args()
    return args


args = get_argument_parse()
