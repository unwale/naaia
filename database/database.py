from psycopg import connect
from config.config import DB_CONFIG
from typing import List
from parsers import NewsItem

class Database:
    def __init__(self):
        self.conn = connect(**DB_CONFIG)
        self.cur = self.conn.cursor()

    def insert_news(self, items: List[NewsItem]) -> None:

        insert_query = """
        INSERT INTO news (raw_title, raw_text, created_at, source_id, url, platform, tags)
        VALUES (%s, %s, %s, %s, %s, %s, %s) 
        """
        
        self.cur.executemany(
            insert_query, 
            [(item.title, item.text, item.date, item.source.name, item.news_url, item.source.platform.name, item.tags) for item in items]
            )
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()
