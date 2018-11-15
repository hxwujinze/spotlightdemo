'''
Main program parsing arguments and running commands
'''

import argparse
import os
import shutil
import sys
import logging
from model import *
from util import *
from commands import *

commands = ['config', 'train', 'batched', 'rf', 'test',
            'examine', 'vis', 'clean', 'pretrain']
models = ['Attn', 'Spotlight']


class Config:
    r"""Command ``config``.
    """

    def __init__(self, parser):
        subs = parser.add_subparsers(title='models available', dest='model')
        subs.required = True
        group_options = set()
        for model in models:
            sub = subs.add_parser(model, formatter_class=parser_formatter)
            group = sub.add_argument_group('setup')
            Model = get_class(model)
            Model.add_arguments(group)
            for action in group._group_actions:
                group_options.add(action.dest)

            def save(args):
                for file in os.listdir(args.workspace):
                    if file.endswith('.json'):
                        os.remove(os.path.join(args.workspace, file))
                model = args.model
                Model = get_class(model)
                setup = {name: value for (name, value) in args._get_kwargs()
                         if name in group_options}
                setup = namedtuple('Setup', setup.keys())(*setup.values())
                conf = os.path.join(args.workspace,
                                    str(model) + '.json')
                m = Model(setup)
                print('model: %s, setup: %s' % (model, str(m.args)))
                save_config(m, conf)

            sub.set_defaults(func=save)

    def run(self, args):
        pass


class Clean:
    def __init__(self, parser):
        parser.add_argument('--all', action='store_true',
                            help='clean the entire workspace')

    def run(self, args):
        if args.all:
            shutil.rmtree(args.workspace)
        else:
            for file in os.scandir(os.path.join(args.workspace, 'snapshots')):
                os.remove(file.path)


class Train:
    r"""Command ``train``.
    """

    def __init__(self, parser):
        parser.add_argument('-r', '--split_frac', type=float, default=0.9,
                            help='train/test split fraction')
        parser.add_argument('-n', '--norm', type=float, default=0.05,
                            help='normalization factor')
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')
        parser.add_argument('-bz', '--batch_size', type=int, default=16,
                            help='batch size')
        parser.add_argument('-d', '--dataset', help='dataset', required=True,
                            choices=['formula', 'melody', 'multiline'])
        parser.add_argument('-s', '--snapshot', help='model snapshot')
        parser.add_argument('-f', '--focus', help='focus module snapshot')
        parser.add_argument('-m', '--spotlight_model', default='none',
                            choices=['markov', 'rnn', 'none'],
                            help='spotlight model')
        parser.add_argument('--refine_each_epoch', action='store_true',
                            help='do spotlight refinement at the end of each'
                            'epoch')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        train(model, args)


class Pretrain:
    def __init__(self, parser):
        parser.add_argument('-r', '--split_frac', type=float, default=0.9,
                            help='train/test split fraction')
        parser.add_argument('-n', '--norm', type=float, default=0.005,
                            help='normalization factor')
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')
        parser.add_argument('-bz', '--batch_size', type=int, default=16,
                            help='batch size')
        parser.add_argument('-d', '--dataset', help='dataset',
                            choices=['formula', 'melody', 'multiline'])
        parser.add_argument('-s', '--snapshot', help='model snapshot')
        parser.add_argument('-f', '--focus', help='focus module snapshot')
        parser.add_argument('-m', '--spotlight_model', default='none',
                            choices=['markov', 'rnn', 'none'],
                            help='spotlight model')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        pretrain_attention(model, args)


class Batched:
    r"""Command ``batched``.
    """

    def __init__(self, parser):
        parser.add_argument('-r', '--split_frac', type=float, default=0.9,
                            help='train/test split fraction')
        parser.add_argument('-n', '--norm', type=float, default=0.005,
                            help='normalization factor')
        parser.add_argument('-N', '--epochs', type=int, default=10,
                            help='number of epochs to train')
        parser.add_argument('-bz', '--batch_size', type=int, default=64,
                            help='batch size')
        parser.add_argument('-d', '--dataset', help='dataset', required=True,
                            choices=['formula', 'melody', 'multiline'])

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        train_batched(model, args)


class Test:
    def __init__(self, parser):
        parser.add_argument('-r', '--split_frac', type=float, default=0.9,
                            help='train/test split fraction')
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')
        parser.add_argument('-d', '--dataset', help='dataset', required=True,
                            choices=['formula', 'melody', 'multiline'])
        parser.add_argument('-f', '--focus', help='focus module')
        parser.add_argument('-m', '--spotlight_model', default='rnn',
                            choices=['markov', 'rnn'],
                            help='spotlight model')
        parser.add_argument('--lcs', action='store_true', help='lcs')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        test(model, args)


class Examine:
    def __init__(self, parser):
        parser.add_argument('-s', '--snapshot',
                            help='model snapshot to test with')
        parser.add_argument('-d', '--dataset', default='full',
                            choices=['sample', 'short', 'full', 'melody'],
                            help='dataset')
        parser.add_argument('-f', '--focus', help='focus module')
        parser.add_argument('-m', '--spotlight_model', default='rnn',
                            choices=['markov', 'rnn'],
                            help='spotlight model')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        examine(model, args)


class Rf:
    def __init__(self, parser):
        parser.add_argument('-s', '--snapshot',
                            help='pretrained model snapshot')
        parser.add_argument('-d', '--dataset', default='full',
                            choices=['sample', 'short', 'full', 'melody'],
                            help='dataset')
        parser.add_argument('--learner', default='PGD',
                            choices=['PGD', 'DQN'],
                            help='algorithm to use')
        parser.add_argument('-t', '--times', type=int, default=10,
                            help='number of times to train')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        reinforce(model, args)


class Vis:
    def __init__(self, parser):
        parser.add_argument('-s', '--snapshot',
                            help='pretrained model snapshot')
        parser.add_argument('-i', '--input', required=True,
                            help='input an image')
        parser.add_argument('-f', '--focus', help='focus module')
        parser.add_argument('-m', '--spotlight_model', default='rnn',
                            choices=['markov', 'rnn'],
                            help='spotlight model')
        parser.add_argument('-W', type=int, default=256, help='width')
        parser.add_argument('-H', type=int, default=128, help='height')
        parser.add_argument('--agent_hs', type=int, default=128,
                            help='agent hidden size')
        parser.add_argument('--colored', action='store_true',
                            help='preserve color')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        visualize(model, args)


parser_formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=parser_formatter)
parser.add_argument('-w', '--workspace',
                    help='workspace dir', default='ws/test')
subparsers = parser.add_subparsers(title='supported commands', dest='command')
subparsers.required = True


def main():
    for command in commands:
        sub = subparsers.add_parser(command, formatter_class=parser_formatter)
        subcommand = get_class(command)(sub)
        sub.set_defaults(func=subcommand.run)

    args = parser.parse_args()
    workspace = args.workspace
    try:
        os.makedirs(os.path.join(workspace, 'snapshots'))
        os.makedirs(os.path.join(workspace, 'results'))
        os.makedirs(os.path.join(workspace, 'logs'))
    except OSError:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logFormatter = ColoredFormatter('%(levelname)s %(asctime)s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fileFormatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

    if args.command != 'config':
        fileHandler = logging.FileHandler(os.path.join(workspace, 'logs',
                                                       args.command + '.log'))
        fileHandler.setFormatter(fileFormatter)
        logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.warn('canceled by user')
    except Exception as e:
        import traceback
        sys.stderr.write(traceback.format_exc())
        logging.error('exception occurred: %s', e)


def get_class(name):
    return globals()[name[0].upper() + name[1:]]


if __name__ == '__main__':
    main()
