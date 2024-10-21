from parsers import NewsSource, Platform
from parsers.telegram import TelegramParser
from parsers.spbu import SpbuParser
from parsers.vk import VkParser

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

        case 'spbu-website':
            return SpbuParser(source)
        
        case _:
            raise Exception(f'Unknown website ID: {source.id}')
        

