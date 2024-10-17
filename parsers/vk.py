import aiohttp
from typing import List
from parsers import Parser, NewsSource, NewsItem

class VkParser(Parser):

    def __init__(self, source: NewsSource):
        super().__init__(source)

    async def fetch_news(self, item_limit: int) -> List[NewsItem]:
        pass