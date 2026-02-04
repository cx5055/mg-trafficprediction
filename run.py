import torch
import numpy as np
import os
import configparser
import argparse
from datetime import datetime
from lib.dataloader_pop import get_dataloader
from ourmodel.model import trapre_model
from lib.multitask_loss import MultiTask_AngleLoss
from ourmodel.trainer import Trainer

def get_args_and_config():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', default='all_traffic', type=str, choices=['all_traffic'], help='only all_traffic')
    pre_parser.add_argument('--model', default='multi_pre', type=str)
    pre_args, _ = pre_parser.parse_known_args()

    config_file = './config_file/{}_{}.conf'.format(pre_args.dataset, pre_args.model)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser(description='arguments')
    parser.set_defaults(dataset=pre_args.dataset, model=pre_args.model)

    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--seq_len', default=config['data']['seq_len'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--ntime', default=config['data']['ntime'], type=int)
    parser.add_argument('--single', default=config.getboolean('data', 'single'), help='single-step')
    parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'], help='val ratio')
    parser.add_argument('--test_ratio', type=float, default=config['data']['val_ratio'], help='test ratio')
    parser.add_argument('--use_pop', default=config['data']['use_pop'], type=eval)
    parser.add_argument('--use_dis', default=config['data']['use_dis'], type=eval)

    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--time_dim', default=config['model']['time_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)

    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--lambda_consist', type=float, default=0.1, help='loss weight')

    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)

    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--debug', default='False', type=eval)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--log_dir', default='./', type=str)

    print(parser.parse_args())
    return parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_args_and_config()
    set_random_seed(args.seed)
    model = trapre_model(args)
    print(model)
    loss = MultiTask_AngleLoss(lambda_consist=args.lambda_consist, eps=0)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )

    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_steps,
            gamma=args.lr_decay_rate,
        )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(current_dir, 'logs', args.dataset, args.model, current_time)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    train_loader, val_loader, test_loader, scaler = get_dataloader(args)
    trainer = Trainer(
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        args,
        lr_scheduler=lr_scheduler,
    )
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        log_dir = 'logs/all_traffic/multi_pre/'
        ckpt_path = os.path.join(log_dir, 'best_test_model.pth')
        model.load_state_dict(torch.load(ckpt_path))
        trainer.test_only(model, trainer.args, test_loader, trainer.logger)
    else:
        raise ValueError('mode must be train or test')


if __name__ == '__main__':
    main()
