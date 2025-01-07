import numpy as np
from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta

precision_to_date_interval = {
    "День": relativedelta(days=2),
    "Неделя": relativedelta(weeks=1),
    "Сезон": relativedelta(months=3),
    "Месяц": relativedelta(months=1),
    "Год": relativedelta(years=1),
}


def relativedelta_to_seconds(rd):
    seconds_in_minute = 60
    seconds_in_hour = 60 * seconds_in_minute
    seconds_in_day = 24 * seconds_in_hour
    seconds_in_month = 30.44 * seconds_in_day
    seconds_in_year = 365.25 * seconds_in_day

    total_seconds = (
        rd.years * seconds_in_year
        + rd.months * seconds_in_month
        + rd.days * seconds_in_day
        + rd.hours * seconds_in_hour
        + rd.minutes * seconds_in_minute
        + rd.seconds
    )
    return total_seconds


def compute_temporal_scores(publication_dates, query, temporal_model):
    """
    Compute temporal scores for a given query and publication dates

    Args:
        publication_dates: list of POSIX timestamps objects
        query: str
        temporal_model: sklearn estimator

    Returns:
        np.array: temporal scores
    """

    publication_dates = np.array(publication_dates)
    temporal_score = np.zeros(publication_dates.shape)
    dates = search_dates(query)
    if dates is None:
        return temporal_score
    for date in dates:
        date_text = date[0]
        words = query.split()

        index = words.index(date_text.split()[0])
        start = max(0, index - 5)
        end = min(len(words), index + 5 + 1)

        window = " ".join(words[start:end])
        precision = temporal_model.predict([window])[0]
        distr_center = (
            date[1] + precision_to_date_interval[precision] / 2
        ).timestamp()
        temporal_score += np.exp(
            -np.square(publication_dates - distr_center)
            / (
                2
                * np.square(
                    relativedelta_to_seconds(
                        precision_to_date_interval[precision]
                    )
                )
            )
        )

    return temporal_score
