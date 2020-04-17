"""Preprocess raw news data scraped by crawler"""

import numpy as np
import ujson as json

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertTokenizer
from args import get_setup_args

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, 'w') as fh:
        json.dump(obj, fh)

def build_features(args, data, y, tokenizer, out_file, data_type):
    print(f'Building features in {data_type} set...')

    # Load data
    titles, contents, img_paths, ids = data[:, 1], data[:, 2], data[:, 3], data[:, 6]

    title_limit = args.title_max_len
    content_limit = args.content_max_len
    input_idxs = []
    atten_masks = []

    num_data = len(data)
    with tqdm(total=num_data) as pbar:
        for title, content in zip(titles, contents):
            tokenized_title = tokenizer.tokenize(title)
            tokenized_content = tokenizer.tokenize(content)

            def truncate_sen(sen, limit):
                return sen if len(sen) <= limit else sen[:limit]

            tokenized_title = truncate_sen(tokenized_title, title_limit)
            tokenized_content = truncate_sen(tokenized_content, content_limit)

            encode_dict = tokenizer.encode_plus(tokenized_title, tokenized_content,
                                                max_length=args.input_len,
                                                pad_to_max_length=True,
                                                is_pretokenized=True)
            input_idx = encode_dict['input_ids']
            input_idxs.append(input_idx)
            atten_mask = encode_dict['attention_mask']
            atten_masks.append(atten_mask)
            pbar.update(1)
    
    print(f'Save {num_data} {data_type} set')
    np.savez(out_file,
             input_idxs=np.array(input_idxs),
             atten_masks=np.array(atten_masks),
             img_paths=img_paths,
             ids=ids.astype(int),
             y=np.array(y))
def pre_process(args):
    # Load data
    data = np.load(args.raw_data_file, allow_pickle=True)[args.times]
    
    # Convert categories to target y idxs
    categories = data[:, 4]
    cat2idx = {cat: idx for idx, cat in enumerate(set(categories))}
    y = [cat2idx[category] for category in categories]

    # Split data into train, dev, test set
    data_train, data_dev, y_train, y_dev = train_test_split(data, y,
                                                            train_size=args.train_size,
                                                            random_state=args.seed,
                                                            shuffle=True,
                                                            stratify=y)
    args.dev_size = args.dev_size / (1 - args.train_size)
    data_dev, data_test, y_dev, y_test = train_test_split(data_dev, y_dev,
                                                          train_size=args.dev_size,
                                                          random_state=args.seed,
                                                          stratify=y_dev)

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    build_features(args, data_train, y_train, tokenizer, args.train_record_file, 'train')
    build_features(args, data_dev, y_dev, tokenizer, args.dev_record_file, 'dev')
    build_features(args, data_test, y_test, tokenizer, args.test_record_file, 'test')

    save(args.cat2idx_file, cat2idx, 'category to index dictionary')

if __name__ == '__main__':
    pre_process(get_setup_args())