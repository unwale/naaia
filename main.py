import os
import dotenv
import asyncio
from parsers import NewsSource, NewsItem, Platform
from typing import List
from telethon import TelegramClient
from utils.parsing import get_parser_for

dotenv.load_dotenv()

sources = [
    NewsSource('Матмех СПбГУ', Platform.TELEGRAM, 'mmspbu'),
    NewsSource('Что там в СПбГУ', Platform.TELEGRAM, 'spbuniversity1724'),
    NewsSource('Профком Матмеха СПбГУ', Platform.TELEGRAM, 'mmprofkomspbu'),
    NewsSource('Лупа и Пупа', Platform.VK, 'lyandpy'),
    NewsSource('СНО СПбГУ', Platform.VK, 'sno.spbu'),
    NewsSource('Новости | СПбГУ', Platform.WEB, 'spbu-official')
]

client = TelegramClient(
    session='/tmp/session', 
    api_id=os.getenv('TELEGRAM_APP_ID'), 
    api_hash=os.getenv('TELEGRAM_APP_HASH')
)

async def main():
    await client.start()

    for source in sources:
        parser = get_parser_for(source)

        if (source.platform == Platform.TELEGRAM):
            parser.client = client

        news: List[NewsItem] = await parser.fetch_news(item_limit=1)
        print(f'{len(news)} news from {source.name}')

    await client.disconnect()    

asyncio.run(main())




