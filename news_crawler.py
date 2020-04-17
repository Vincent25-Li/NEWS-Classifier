import os, csv, re, time
from datetime import datetime

import ujson as json
import requests
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

from args import get_crawler_args

def get_page(url):
    '''Get the page content

    Args:
        url (str): url to scrape

    Returns:
        response (obj): object of web content
    '''
    time.sleep(6)
    headers = {'user-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0'}
    response = requests.get(url, headers=headers, timeout=5)
    if response.status_code == 200:
        return response
    else:
        print(f'Something went wrong when scraping {url}')
        return None


def copy_page(url):
    '''Copy the web content into local disk

    Args:
        url (str): url of web page
    '''
    response = get_page(url)
    with open('check.html', 'wb') as f:
        f.write(response.content)
        

class ScrapeLiberty():
    """Download Liberty times news
    
    Functions:
        scrape: scrape the news in categories
    """
    def __init__(self, data=None):
        self.url_base = 'https://news.ltn.com.tw/ajax/breakingnews'
        self.categories = {'politics': 'politic',
                           'society': 'society',
                           'life': 'life',
                           'world': 'world',
                           'local': 'local',
                           'novelty': 'novelty',
                           'entertainment': 'entertainment',
                           'sports': 'sport'}
        self.news = 'liberty'
        self.data = data
        self.id = len(data) if data else 0
        self.titles = self.data[:, 2] if np.any(self.data) else []
    
    def scrape(self, s_category=None):
        data = []
        for category in self.categories.keys():
            if s_category is not None and category != s_category: continue

            index = 1
            
            while True:
                url = f'{self.url_base}/{category}/{index}'
                try:
                    news_list = get_page(url)
                except:
                    index += 1
                    continue
                news_list = json.loads(news_list.text)['data']
                if news_list == []: break
                news_list = news_list if isinstance(news_list, list) else news_list.values()
                
                with tqdm(total=len(news_list)) as pbar:
                    for news in news_list:
                        try:
                            datum = self.process_page(news, category)
                            self.id += 1
                        except: datum = None
                        if datum:
                            data.append(datum)
                        pbar.update(1)
                        pbar.set_postfix(index=index, category=self.categories[category], news=self.news)
                index += 1
        
        if self.data is None or np.any(self.data) is None:
            self.data = np.array(data)
        else:
            self.data = np.concatenate([np.array(data), self.data], axis=0)
    
    def process_page(self, news, category):
        url = news['url']
        response = get_page(url)
        soup = BeautifulSoup(response.text, 'lxml')
        article = self.get_article(soup, category)

        title = self.get_title(article)
        if title in self.titles:
            return None

        date = self.get_date(article, category)
        content = self.get_content(article, category)
        image_path = self.get_image(news['photo_S'])

        return [date, title, content, image_path, self.categories[category], url, self.id]
    
    def get_article(self, soup, category):
        if category in ['entertainment', 'sports']:
            return soup.find('div', class_="content")
        else:
            return soup.find(itemprop="articleBody")

    def get_date(self, article, category):
        if category in ['entertainment', 'sports']:
            pattern = '\d{4}\/\d{2}\/\d{2}'
            date = re.search(pattern, article.text).group()
            date = datetime.strptime(date, '%Y/%m/%d').date()
        else:
            pattern = '\d{4}-\d{2}-\d{2}'
            date = re.search(pattern, article.text).group()
            date = datetime.strptime(date, '%Y-%m-%d').date()
        return date

    def get_title(self, article):
        return article.h1.text

    def get_content(self, article, category):
        if category in ['entertainment']:
            news_content_list = article.find_all('p', class_='')
            content = ''.join([content.text for content in news_content_list if not content.span])
        elif category in ['sports']:
            news_content_list = article.find_all('p', class_='')
            content = ''.join([content.text for content in news_content_list if not content.img])
        else:
            news_content_list = article.find('div', class_='text boxTitle boxText').find_all('p', class_='', recursive=False)
            content = ''.join([content.text for content in news_content_list])
        return content
    
    def get_image(self, url):
        image_dir = os.path.join('data', 'pictures', self.news)
        image_name = f'{len(os.listdir(image_dir))}.jpg'
        image_path = os.path.join(image_dir, image_name)
        image = get_page(url)
        with open(os.path.join(os.getcwd(), image_path), 'wb') as f:
            for chunk in image:
                f.write(chunk)
        return image_path


class ScrapeChina(ScrapeLiberty):
    """Download China times news
    
    Functions:
        scrape: scrape the news in categories
    """
    def __init__(self, data=None):
        ScrapeLiberty.__init__(self, data)
        self.url_base = 'https://www.chinatimes.com'
        self.categories = {'star': 'entertainment',
                           'politic': 'politic',
                           'sports': 'sport',
                           'society': 'society'}
        self.news = 'china'
    
    def scrape(self, s_category=None):
        data = []
        for category in self.categories.keys():
            if s_category is not None and category != s_category: continue
            for index in range(1, 11):  
                url = f'{self.url_base}/{category}/total?page={index}&chdtv'
                try:
                    response = get_page(url)
                except:
                    continue
                soup = BeautifulSoup(response.text, 'lxml')
                news_list = soup.find('div', class_="container") \
                                .find('ul', class_='vertical-list') \
                                .find_all('li')
                
                with tqdm(total=len(news_list)) as pbar:
                    for news in news_list:
                        try:
                            datum = self.process_page(news, category)
                        except: datum = None
                        if datum:
                            data.append(datum)
                        pbar.update(1)
                        pbar.set_postfix(index=index, category=self.categories[category], news=self.news)
        
        if self.data is None or np.any(self.data) is None:
            self.data = np.array(data)
        else:
            self.data = np.concatenate([np.array(data), self.data], axis=0)
    
    def process_page(self, news, category):
        url = f'{self.url_base}{news.h3.a["href"]}'
        response = get_page(url)
        soup = BeautifulSoup(response.text, 'lxml')
        article = self.get_article(soup)

        title = self.get_title(article).replace('\xa0', ' ').replace('\u3000', ' ')
        if title in self.titles:
            return None

        date = self.get_date(article, category)
        content = self.get_content(article, category)
        image_path = self.get_image(news.img["src"])

        return [date, title, content, image_path, self.categories[category], url]
    
    def get_article(self, soup):
        return soup.article
    
    def get_date(self, article, category):
        pattern = '\d{4}\/\d{2}\/\d{2}'
        date = re.search(pattern, article.time.text).group()
        date = datetime.strptime(date, '%Y/%m/%d').date()
        return date

    def get_content(self, article, category):
        news_content_list = article.find('div', class_='article-body').find_all('p')
        content = ''.join([content.text for content in news_content_list])
        return content


class ScrapeUDN(ScrapeLiberty):
    """Download UDN news
    
    Functions:
        scrape: scrape the news in categories
    """
    def __init__(self, data=None):
        ScrapeLiberty.__init__(self, data)
        self.url_base = 'https://udn.com'
        self.url_base_s = 'https://stars.udn.com'
        self.categories = {'stock': (6645, 90),
                           'sport': (7227, 193),
                           'society': (6639, 123),
                           'entertainment': 'stars'}
        self.news = 'udn'
    
    def scrape(self, s_category=None):
        data = []
        for category in self.categories.keys():
            if s_category is not None and category != s_category: continue
            index = 0
            counts = 0
            counts_e = 0
            while True: 
                url = self.get_category_url(category, index)
                try:
                    response = get_page(url)
                except:
                    index += 1
                    continue
                news_list = self.get_news_list(response, category)
                if not news_list: break
                
                with tqdm(total=len(news_list)) as pbar:
                    for news in news_list:
                        try:
                            datum = self.process_page(news, category)
                        except: datum = None
                        if datum:
                            data.append(datum)
                            if category == 'entertainment': 
                                counts_e += 1
                            else:
                                counts += 1
                        pbar.update(1)
                        pbar.set_postfix(index=index, category=category, news=self.news)
                if counts_e > (counts / 3): break
                index += 1
        
        if self.data is None or np.any(self.data) is None:
            self.data = np.array(data)
        else:
            self.data = np.concatenate([np.array(data), self.data], axis=0)
    
    def process_page(self, news, category):
        url = self.get_article_url(news, category)
        response = get_page(url)
        soup = BeautifulSoup(response.text, 'lxml')
        article = self.get_article(soup, category)

        title = self.get_title(article).replace('\u3000', ' ')
        if title in self.titles:
            return None

        date = self.get_date(article)
        content = self.get_content(article, category)
        img_url =  news.find('a', class_='item-image').img['data-original'] \
                    if category == 'entertainment' else news['url']
        image_path = self.get_image(img_url)

        return [date, title, content, image_path, category, url]
    
    def get_category_url(self, category, index):
        if category == 'entertainment':
            url = f'{self.url_base_s}/common/ajax_show_more/news/{index + 1}/0/0/0'
        else:
            (cate_id, totalRecNo) = self.categories[category]
            url_remaining = f'/api/more?page={index}&channelId=2&type=cate_latest_news&cate_id={cate_id}&totalRecNo={totalRecNo}'
            url = self.url_base + url_remaining
        return url
    
    def get_news_list(self, response, category):
        if category == 'entertainment':
            response = json.loads(response.text)
            soup = BeautifulSoup(response['_html'], 'lxml')
            news_list = soup.find_all('div', class_='item')
        else:
            news_list = json.loads(response.text)
            news_list = news_list['lists'] if 'lists' in news_list else None
        return news_list
    
    def get_article_url(self, news, category):
        if category == 'entertainment':
            url = f'{self.url_base_s}{news.find("div", class_="item-text").a["href"]}'
        else:
            url = f'{self.url_base}{news["titleLink"]}'
        return url
    
    def get_article(self, soup, category):
        if category == 'entertainment':
            article = soup.find('section', class_='cate-article')
        else:
            article = soup.find('section', class_='article-content__wrapper')
        return article
    
    def get_date(self, article):
        pattern = '\d{4}-\d{2}-\d{2}'
        date = re.search(pattern, article.text).group()
        date = datetime.strptime(date, '%Y-%m-%d').date()
        return date

    def get_content(self, article, category):
        if category == 'entertainment':
            news_content_list = article.find('div', class_='article').find_all('p')
        else:
            news_content_list = article.find('div', class_='article-content__paragraph').find_all('p')
            
        content = ''.join([content.text for content in news_content_list])
        content = content.replace('\r', '').replace('\n', '')
        return content


def main(args):
    if os.path.exists(args.crawler_file):
        data = np.load(args.crawler_file, allow_pickle=True)
        crawlers = {'liberty': ScrapeLiberty(data['liberty']),
                    'china': ScrapeChina(data['china']),
                    'udn': ScrapeUDN(data['udn'])}
    else:
        crawlers = {'liberty': ScrapeLiberty(),
                    'china': ScrapeChina(),
                    'udn': ScrapeUDN()}

    if isinstance(args.targets, str):
        print(f'Scraping {args.targets}...')
        crawlers[args.targets].scrape(args.s_category)
    else:
        for target in args.targets:
            print(f'Scraping {target}...')
            crawlers[target].scrape()
    
    np.savez(args.crawler_file,
             liberty=crawlers['liberty'].data,
             china=crawlers['china'].data,
             udn=crawlers['udn'].data)
    

if __name__ == '__main__':
    main(get_crawler_args())