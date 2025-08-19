import argparse


def get_argument_parse():
    parse = argparse.ArgumentParser(description='parameters')
    parse.add_argument('--trial_run', type=int, default=2, help='Experiment number of the trial run.')
    parse.add_argument('--dataset', type=str, default='SZU_R1', choices=['DFC Houston 2018', 'SZU_R1'])
    parse.add_argument('--lr', type=float, default=1e-3, help='learning rate for noise predictor')
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--seed', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=30)
    parse.add_argument('--log_dir', default='../tf-logs')
    args = parse.parse_args()
    # 转换 betas 字符串为浮点数元组
    try:
        args.betas = tuple(map(float, args.betas.split(',')))
        if len(args.betas) != 2:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError(
            "betas must be two comma-separated floats (e.g. '0.5,0.999')")
    return args


args = get_argument_parse()
