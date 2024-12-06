from datetime import datetime

from . import NewsSource, Platform
from .spbu import SpbuParser
from .telegram import TelegramParser
from .vk import VkParser


def get_parser_for(source: NewsSource):

    match (source.platform):

        case Platform.TELEGRAM:
            return TelegramParser(source=source)

        case Platform.VK:
            return VkParser(source=source)

        case Platform.WEB:
            return _get_web_scraper(source=source)


def _get_web_scraper(source: NewsSource):
    match (source.id):

        case "spbu-website":
            return SpbuParser(source)

        case _:
            raise Exception(f"Unknown website ID: {source.id}")


def datestr_to_datetime(date_str: str) -> datetime:
    # russian to english months mapping
    months = {
        "января": "January",
        "февраля": "February",
        "марта": "March",
        "апреля": "April",
        "мая": "May",
        "июня": "June",
        "июля": "July",
        "августа": "August",
        "сентября": "September",
        "октября": "October",
        "ноября": "November",
        "декабря": "December",
    }

    for rus_month, eng_month in months.items():
        date_str = date_str.replace(rus_month, eng_month)

    return datetime.strptime(date_str.strip(), "%d %B %Y")
