import asyncio
import logging
import os
from dataclasses import asdict
from typing import List

import pandas as pd
from dotenv import load_dotenv
from scrapers import NewsItem, NewsSource, Platform
from scrapers.utils import get_parser_for
from telethon import TelegramClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper")
load_dotenv("../.env")

sources = [
    NewsSource("Матмех СПбГУ", Platform.TELEGRAM, "mmspbu"),
    NewsSource("Что там в СПбГУ", Platform.TELEGRAM, "spbuniversity1724"),
    NewsSource("Профком Матмеха СПбГУ", Platform.TELEGRAM, "mmprofkomspbu"),
    NewsSource("Студенческий совет СПбГУ", Platform.VK, "ssspbu"),
    NewsSource("Факультет МКН СПбГУ", Platform.VK, "spbumathcs"),
    NewsSource("Лупа и Пупа", Platform.VK, "lyandpy"),
    NewsSource("СНО СПбГУ", Platform.VK, "sno.spbu"),
    NewsSource("СПбГУ", Platform.VK, "spb1724"),
    NewsSource("Новости | СПбГУ", Platform.WEB, "spbu-website"),
]

data: pd.DataFrame = pd.DataFrame(
    columns=[
        "date",
        "title",
        "summary",
        "text",
        "source_name",
        "source_id",
        "source_platform",
        "news_url",
        "tags",
    ]
)

client = TelegramClient(
    session="./tmp/session",
    api_id=os.getenv("TELEGRAM_APP_ID"),
    api_hash=os.getenv("TELEGRAM_APP_HASH"),
)


async def main():
    global data

    await client.start()

    for source in sources:
        logger.info(f"Creating a {source.platform.name} parser")
        parser = get_parser_for(source)

        if source.platform == Platform.TELEGRAM:
            parser.client = client
        if source.platform == Platform.WEB:
            parser.max_requests = 5
        logger.info("Parsing")
        news: List[NewsItem] = await parser.fetch_news()

        # make news match the data schema
        # and remove the platform object
        news = [asdict(item) for item in news]
        for item in news:
            item["source_name"] = source.name
            item["source_id"] = source.id
            item["source_platform"] = source.platform.name
            item.pop("source")

        data = pd.concat([data, pd.DataFrame(news)], ignore_index=True)

        logger.info(f"{len(news)} news parsed from {source.name}")

    await client.disconnect()

    data.to_json("../data/raw/data.jsonl", orient="records", lines=True)
    logger.info("Data saved to ../data/raw/data.jsonl")


asyncio.run(main())
