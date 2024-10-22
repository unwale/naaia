from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List

class Platform(Enum):
    TELEGRAM=1
    VK=2
    WEB=3

@dataclass
class NewsSource:
    """
    Represents a news source for parsing

    Attributes:
        name : str
            Screen name
        platform : Platform
        id : str
            Identificator which the source can be distinguished by within the same platform 
            For all platforms except web the platform's id is used
            examples:
                mmspbu - Telegram channel ID 
                lyandpy - VK group ID,
                spbu-official - official SPbU news page 
    """

    name: str
    platform: Platform
    id: str

@dataclass
class NewsItem:
    date: datetime
    title: str
    text: str
    source: NewsSource
    news_url: str


class Parser(ABC):

    def __init__(self, source: NewsSource) -> None:
        self.max_limit: int = 0
        self.max_requests: int = 10
        self.source: NewsSource = source
    
    @abstractmethod
    async def fetch_news(self, item_limit: int) -> List[NewsItem]:
        """
        Fetches news from given news source

        Parameters:
        item_limit : int
            Maximum count of news to fetch

        Returns:
            List of NewsItem objects, each having source set as the source provided to the parser
        """
        pass

