import os
import aiohttp
import json
from typing import List
from urllib.parse import urlencode
from parsers import Parser, NewsSource, NewsItem
from datetime import datetime

class VkParser(Parser):

    base_url = 'https://api.vk.ru/method/{method}?{params}'

    def __init__(self, source: NewsSource):
        super().__init__(source)

    async def fetch_news(self, item_limit: int) -> List[NewsItem]:
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
                request_url = self.base_url.format(method='wall.get', params = urlencode(params))
                response = await session.get(request_url)
                response_json: dict = json.loads(await response.text())['response']
                if not ('items' in response_json.keys()): continue
                for post in response_json['items']:
                    if post['marked_as_ads']: continue
                    news.append(
                        NewsItem(
                            date = datetime.fromtimestamp(post['date']),
                            text = post['text'],
                            source=self.source
                        )
                    )
        return news