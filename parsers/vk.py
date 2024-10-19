import os
import aiohttp
import json
from typing import List
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from parsers import Parser, NewsSource, NewsItem
from datetime import datetime

class VkParser(Parser):

    base_url = 'https://api.vk.ru/method/{method}?{params}'

    def __init__(self, source: NewsSource):
        super().__init__(source)
        params = {
                    'access_token': os.getenv('VK_API_TOKEN'),
                    'domain': self.source.id,
                    'count': 1,
                    'v': 5.199
                }
        req = Request(self.base_url.format(method = 'wall.get', params = urlencode(params)))
        with urlopen(req) as response:
            self.max_limit = json.loads(response.read())['response']['count']

    async def fetch_news(self, item_limit: int = None) -> List[NewsItem]:
        if item_limit is None:
            item_limit = self.max_limit

        news = []
        async with aiohttp.ClientSession() as session:
            for i in range((item_limit // 100) + 1):
                params = {
                    'access_token': os.getenv('VK_API_TOKEN'),
                    'domain': self.source.id,
                    'count': min(100, item_limit - i*100),
                    'offset': i*100,
                    'v': 5.199
                }
                request_url = self.base_url.format(method = 'wall.get', params = urlencode(params))
                response = await session.get(request_url)
                response_json: dict = json.loads(await response.text())['response']
                if not ('items' in response_json.keys()): continue
                for post in response_json['items']:
                    if post['marked_as_ads']: continue
                    news.append(
                        NewsItem(
                            date = datetime.fromtimestamp(post['date']),
                            title = None,
                            text = post['text'],
                            source = self.source
                        )
                    )
        return news