from datetime import datetime

def datestr_to_datetime(date_str: str) -> datetime:
    # Russian to english months mapping 
    months = {
        'января': 'January', 'февраля': 'February', 'марта': 'March', 'апреля': 'April',
        'мая': 'May', 'июня': 'June', 'июля': 'July', 'августа': 'August',
        'сентября': 'September', 'октября': 'October', 'ноября': 'November', 'декабря': 'December'
    }

    # Replace Russian month with English month
    for rus_month, eng_month in months.items():
        date_str = date_str.replace(rus_month, eng_month)

    return datetime.strptime(date_str.strip(), '%d %B %Y')    