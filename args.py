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

    parser.add_argument('--raw_data_file',
                        type=str,
                        default='./data/train_raw.npz',
                        help='File to precessed')
    parser.add_argument('--times',
                        type=str,
                        default='liberty',
                        help='Determine publisher to process')
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

def add_common_args(parser):
    """Add arguments common to scripts: setup.py, train.py"""
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