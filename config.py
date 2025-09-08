import argparse


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=2, help='Experiment number of the trial run.')
    parse.add_argument('--data_dir', type=str, default='../autodl-fs/dataset/SZUTreeData2.0/SZUTreeData_R1_2.0')
    parse.add_argument('--lr', type=float, default=1e-6, help='learning rate for noise predictor')
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--seed', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=1)
    parse.add_argument('--log_dir', default='../tf-logs')
    args = parse.parse_args()
    return args


args = get_argument_parse()
