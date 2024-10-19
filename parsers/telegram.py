import json
from typing import List
from parsers import Parser, NewsItem, NewsSource
from telethon import TelegramClient


class TelegramParser(Parser):

    def __init__(self, source: NewsSource):
        super().__init__(source)
        self.max_limit = None
        self.client: TelegramClient = None
    
    async def fetch_news(self, item_limit: int = None) -> List[NewsItem]:
        if item_limit is None:
            item_limit = self.max_limit
            
        news = []
        try:
            async for message in self.client.iter_messages(self.source.id, limit=item_limit, wait_time=2):
                item = NewsItem(
                    date = message.date,
                    title = None, 
                    text = message.text, 
                    source = self.source
                    )
                news.append(item)
        except Exception as e:
            print(f"An error occurred: {e}")

        return news
