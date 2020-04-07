"""Command-line arguments for news_crawler.py
"""

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
    
    args = parser.parse_args()

    return args