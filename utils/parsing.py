import os
from parsers import NewsSource, Platform
from parsers.telegram import TelegramParser
from parsers.spbu import SpbuParser

def get_parser_for(source: NewsSource):

    match (source.platform):

        case Platform.TELEGRAM:
            return TelegramParser(source=source)
        
        case Platform.WEB:
            return _get_web_scraper(source=source)
        
        case _:
            ...
            #raise Exception(f'No such platform: {source.platform.name}')


      
            

def _get_web_scraper(source: NewsSource):
    match (source.id):

        case "spbu-official":
            return SpbuParser(2)
        
        case _:
            raise Exception(f'Unknown website ID: {source.id}')
        

