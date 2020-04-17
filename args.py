"""Command-line arguments for news_crawler.py"""

import argparse


def get_crawler_args():
    """Get arguments needed in news_crawler.py"""
    parser = argparse.ArgumentParser('Scrape news of LibertyTimes, ChinaTimes and UDN')

    parser.add_argument('--crawler_file',
                        type=str,
                        default='./data/train_raw.npz',
                        help='File to load or save for the scraped news')
    parser.add_argument('--targets',
                        nargs='+',
                        type=str,
                        default='liberty',
                        help="Target Times to scrape: [liberty, china, udn]")
    parser.add_argument('--s_category',
                        type=str,
                        default=None,
                        help="Specific category to scrape")
    
    args = parser.parse_args()

    return args

def get_setup_args():
    """Get arguments needed in setup.py"""
    parser = argparse.ArgumentParser('Preprocess raw news data')

    add_common_args(parser)

    parser.add_argument('--title_max_len',
                        type=int,
                        default=39,
                        help='Maximun length of tokenized title')
    parser.add_argument('--content_max_len',
                        type=int,
                        default=470,
                        help='Maximun length of tokenized content')
    parser.add_argument('--input_len',
                        type=int,
                        default=512,
                        help='Length of the model input')
    parser.add_argument('--cat2idx_file',
                        type=str,
                        default='./data/cat2idx.json')
    parser.add_argument('--train_size',
                        type=float,
                        default=0.90,
                        help='Proportion of data in training set')
    parser.add_argument('--dev_size',
                        type=float,
                        default=0.05,
                        help='Proportion of data in development set')
    
    args = parser.parse_args()
    
    return args

def get_train_args():
    """Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Train a model on NEWS')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=2000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--num_labels',
                        type=int,
                        default=8,
                        help='Number of labels for classification.')
    parser.add_argument('--use_img',
                        type=lambda s: s.lower().startwith('t'),
                        default=True,
                        help='Whether to use images for prediction')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name == 'F1':
        # Best checkpoint is the one that maximizes F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args

def add_common_args(parser):
    """Add arguments common to scripts: setup.py, train.py, test.py"""
    parser.add_argument('--raw_data_file',
                        type=str,
                        default='./data/train_raw.npz',
                        help='Original data file')
    parser.add_argument('--times',
                        type=str,
                        default='liberty',
                        help='Determine publisher to process')
    parser.add_argument('--seed',
                        type=int,
                        default=112,
                        help='Random seed for reproducibility')
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')

def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')