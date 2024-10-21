import asyncio
import aiohttp
from lxml import html
from typing import List
from math import ceil
from parsers import Parser, NewsItem, NewsSource
from utils.date import datestr_to_datetime

class SpbuParser(Parser):

    base_url = 'https://spbu.ru'
    page_url = base_url + '/news-events/novosti?page={}'

    def __init__(self, source: NewsSource):
        super().__init__(source)
        self.max_limit = 615 * 10
        self.results = []

    async def fetch_page(self, session, page_number) -> str:
        """
        Fetch a single page asynchronously.

        Parameters:
            session: The aiohttp session object.
            page_number: The page number to fetch.

        Returns:
            HTML content of the fetched page.
        """
        url = self.page_url.format(page_number)
        async with session.get(url) as response:
            return await response.text()

    async def parse_news_urls(self, page_content) -> List[str]:
        """
        Parse a single page's content using BeautifulSoup.

        Parameters:
        - page_content: The raw HTML content of the page.
        
        Returns:
        - List of news page urls 
        """
        tree = html.fromstring(page_content)
        news_urls = tree.xpath('//a[@class="card__header" and not(ancestor::aside)]/@href')
        return news_urls


    async def scrape_page(self, session, page_number) -> None:
        """
        Fetch and parse all news from the page, then store the results.

        Parameters:
        - session: The aiohttp session object.
        - page_number: The number of pages to scrape
        """
        news = []

        page_content = await self.fetch_page(session, page_number)
        news_urls = await self.parse_news_urls(page_content)
        for url in news_urls:
            async with session.get(self.base_url + url) as response:
                tree = html.fromstring(await response.text())
                try:
                    title = tree.xpath('//h1[@class="post__title"]/text()')[0]
                    
                    summary = tree.xpath('//div[@class="post__desc"]/text()')
                    summary = summary[0] if len(summary) > 0 else ''
                    text = '/n'.join([el.text_content() for el in tree.xpath('//article[@class="editor-wrap"]')])

                    date = tree.xpath('//span[@class="post__date"]/text()')
                    if (len(date) == 0): 
                        date = tree.xpath('//span[@class="card__date"]/text()')
                    date = date[0]
                    
                    news.append(
                        NewsItem(
                            date = datestr_to_datetime(date),
                            title = title,
                            text = f'{summary}\n\n{text}',
                            source = self.source
                        )
                    )

                except:
                    print(f'Error parsing {url}') 
        self.results[page_number] = news

    
    async def fetch_news(self, item_limit: int = None) -> List[NewsItem]:
        if item_limit is None:
            item_limit = self.max_limit

        total_pages = ceil(item_limit / 10)
        self.results = [0] * total_pages

        async with aiohttp.ClientSession() as session:
            for i in range(ceil(total_pages / self.max_requests)):
                tasks = []
                num_pages_to_parse = min(self.max_requests, total_pages - i * self.max_requests)

                print(f'[i] - Parsing pages {i*self.max_requests} - {i*self.max_requests + num_pages_to_parse}')
                for request_number in range(0, num_pages_to_parse):
                    page_number = i * self.max_requests + request_number
                    task = asyncio.create_task(self.scrape_page(session, page_number))
                    tasks.append(task)

                await asyncio.gather(*tasks)
                await asyncio.sleep(1)
        return [item for result in self.results for item in result]
