import os
import dotenv
import asyncio
from parsers import NewsSource, NewsItem, Platform
from typing import List
from telethon import TelegramClient
from utils.parsing import get_parser_for
from database.database import Database

dotenv.load_dotenv()

sources = [
    NewsSource('Матмех СПбГУ', Platform.TELEGRAM, 'mmspbu'),
    NewsSource('Что там в СПбГУ', Platform.TELEGRAM, 'spbuniversity1724'),
    NewsSource('Профком Матмеха СПбГУ', Platform.TELEGRAM, 'mmprofkomspbu'),
    NewsSource('Студенческий совет СПбГУ', Platform.VK, 'ssspbu'),
    NewsSource('Факультет МКН СПбГУ', Platform.VK, 'spbumathcs'),
    NewsSource('Лупа и Пупа', Platform.VK, 'lyandpy'),
    NewsSource('СНО СПбГУ', Platform.VK, 'sno.spbu'),
    NewsSource('СПбГУ', Platform.VK, 'spb1724'),
    NewsSource('Новости | СПбГУ', Platform.WEB, 'spbu-website')
]

db = Database()

client = TelegramClient(
    session='/tmp/session', 
    api_id=os.getenv('TELEGRAM_APP_ID'), 
    api_hash=os.getenv('TELEGRAM_APP_HASH')
)

async def main():
    await client.start()

    for source in sources:
        print(f'[i] Creating a {source.platform.name} parser')
        parser = get_parser_for(source)

        if (source.platform == Platform.TELEGRAM):
            parser.client = client
            
        print('[i] Parsing')
        news: List[NewsItem] = await parser.fetch_news()
        print(f'[i] {len(news)} news parsed from {source.name}')
        db.insert_news(news)
        print(f'[i] PostgreSQL transaction completed')

    await client.disconnect()    
    db.close()

asyncio.run(main())




