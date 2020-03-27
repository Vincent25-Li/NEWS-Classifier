import os, csv
import ujson as json

import requests
from bs4 import BeautifulSoup

def get_page(url):
    '''Get the page content

    Parameters
        url (str): url to scrape

    Returns
        response (obj): object of web content
    '''
    headers = {'user-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response
    else:
        print(f'Something went wrong when scraping {url}')

def copy_content(response):
    '''Copy the web content in local disk

    Parameters
        response (obj): object of web content
    '''
    with open('check.html', 'wb') as f:
        f.write(response.content)

def save_data(news_title, news_contents):
    '''Save news_title and news_contents in data file
    
    Parameters
        news_title (str): news title
        news_content (str): news content
    '''
    file_name = 'train.csv'
    file = os.path.join('data', file_name)
    if os.path.exists(file): 
        mode = 'a'
    else:
        mode = 'w'
    # write in the data
    with open(file, mode=mode, newline='') as csv_file:
        fieldnames = ['Title', 'Contents']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == 'w': csv_writer.writeheader()
        data_dict = dict(zip(fieldnames, [news_title, news_contents]))
        csv_writer.writerow(data_dict)

    print(f'{news_title} written in successfully')

def scrape_liberty_times():
    '''Get data from Liberty Times

    Parameters
        soup (obj): soup object of web content

    Returns
    '''
    url = 'https://news.ltn.com.tw/news/society/breakingnews/3092149' 
    response = get_page(url)
    soup = BeautifulSoup(response.text, 'lxml')
    article = soup.find(itemprop="articleBody")
    news_title = article.h1.text
    news_contents_list = article.find('div', class_='text boxTitle boxText').find_all('p', class_='', recursive=False)
    news_contents = ''
    for news_content in news_contents_list:
        news_contents += news_content.text

    print(news_title, news_contents)
    # return news_title, news_contents


def main():
    scrape_liberty_times()

if __name__ == '__main__':
    main()