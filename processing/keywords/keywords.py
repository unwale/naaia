import keybert
import numpy as np
import spacy
from yake import KeywordExtractor
import nltk
from rake_nltk import Rake
from rapidfuzz import fuzz
import pytextrank

spc = spacy.load('ru_core_news_lg')

stopwords = nltk.corpus.stopwords.words('russian')
_doc = spc(' '.join(stopwords))
stopwords = {w.text: [w.pos_] for w in _doc}

spc.add_pipe('textrank', config={'stopwords': stopwords})
yake = KeywordExtractor(lan='ru', n=2, top=15, stopwords=nltk.corpus.stopwords.words('russian'))
rake = Rake(max_length=2, language='russian', stopwords=nltk.corpus.stopwords.words('russian'))


def remove_fuzzy_duplicates(strings, threshold=55):
    unique_strings = []

    for current_string in strings:
        # Check for duplicates based on the threshold
        found_duplicate = False
        for unique_string in unique_strings:
            if fuzz.ratio(current_string, unique_string) >= threshold:
                found_duplicate = True
                # Replace with the longest string
                if len(current_string) > len(unique_string):
                    unique_strings.remove(unique_string)
                    unique_strings.append(current_string)
                break  # No need to check further if we found a duplicate

        # If no duplicate was found, add the current string
        if not found_duplicate:
            unique_strings.append(current_string)

    return unique_strings


def get_keywords_yake(text):
    """
    return yake keywords with normalized scores from 0 to 1
    """
    keywords = yake.extract_keywords(text)
    if not keywords:
        return []
    # apply deduplication with fuzzy matching
    unique_keywords = remove_fuzzy_duplicates([keyword[0] for keyword in keywords])
    result = [(keyword[0], keyword[1]) for keyword in keywords if keyword[0] in unique_keywords]
    # normalize scores
    scores = np.array([keyword[1] for keyword in result])
    scores = scores - scores.max()
    scores = 1 - scores
    return sorted([(keyword[0], score) for keyword, score in zip(result, scores)], key=lambda x: x[1], reverse=True)


def get_keywords_rake(text):
    rake.extract_keywords_from_text(text)



spc = spacy.load('ru_core_news_lg')

stopwords = nltk.corpus.stopwords.words('russian')
_doc = spc(' '.join(stopwords))
stopwords = { w.text: [w.pos_] for w in _doc }

spc.add_pipe('textrank', config={'stopwords': stopwords})
yake = KeywordExtractor(lan='ru', n=2, top=15, stopwords=nltk.corpus.stopwords.words('russian'))
rake = Rake(max_length=2, language='russian', stopwords=nltk.corpus.stopwords.words('russian'))


def remove_fuzzy_duplicates(strings, threshold=55):
    unique_strings = []

    for current_string in strings:
        # Check for duplicates based on the threshold
        found_duplicate = False
        for unique_string in unique_strings:
            if fuzz.ratio(current_string, unique_string) >= threshold:
                found_duplicate = True
                # Replace with the longest string
                if len(current_string) > len(unique_string):
                    unique_strings.remove(unique_string)
                    unique_strings.append(current_string)
                break  # No need to check further if we found a duplicate

        # If no duplicate was found, add the current string
        if not found_duplicate:
            unique_strings.append(current_string)

    return unique_strings

def get_keywords_yake(text):
    """
    return yake keywords with normalized scores from 0 to 1
    """
    keywords = yake.extract_keywords(text)
    if not keywords:
        return []
    # apply deduplication with fuzzy matching
    unique_keywords = remove_fuzzy_duplicates([keyword[0] for keyword in keywords])
    result = [(keyword[0], keyword[1]) for keyword in keywords if keyword[0] in unique_keywords]
    # normalize scores
    scores = np.array([keyword[1] for keyword in result])
    scores = scores - scores.max()
    scores = 1 - scores
    return sorted([(keyword[0], score) for keyword, score in zip(result, scores)], key=lambda x: x[1], reverse=True)

def get_keywords_rake(text):
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases_with_scores()
    return [keyword[1] for keyword in keywords[::15]]

def get_keywords_textrank(text):
    doc = spc(text)
    keywords = {phrase.text: phrase.rank for phrase in doc._.phrases[:10]}
    if not keywords:
        return []
    # apply deduplication with fuzzy matching
    unique_keywords = remove_fuzzy_duplicates(keywords.keys())
    result = [(keyword, keywords[keyword]) for keyword in unique_keywords]
    # normalize scores
    scores = np.array([keyword[1] for keyword in result])
    scores = scores / scores.max()
    return [(keyword[0], score) for keyword, score in zip(result, scores)]

def get_keywords(text, n = 10):
    # uses yake and textrank
    yake_keywords = get_keywords_yake(text)
    textrank_keywords = get_keywords_textrank(text)
    # weight scores
    yake_keywords = [(keyword[0], keyword[1]) for keyword in yake_keywords]
    textrank_keywords = [(keyword[0], keyword[1]) for keyword in textrank_keywords]
    # combine keywords with summing scores if keyword is in both lists and remove fuzzy duplicates

    unique_keywords = {}
    for keyword in textrank_keywords:
        unique_keywords[keyword[0]] = keyword[1] * .65
    for keyword in yake_keywords:
        duplicate_found = False
        for unique_keyword in textrank_keywords:
            if fuzz.ratio(keyword[0], unique_keyword) >= 50:
                unique_keywords[unique_keyword] += keyword[1] * .35
                duplicate_found = True
                break
        if not duplicate_found:    
            unique_keywords[keyword[0]] = keyword[1] * .35
    unique_keywords = sorted(unique_keywords.items(), key=lambda x: x[1], reverse=True)
    return unique_keywords[:n] + [(keyword, score) for keyword, score in unique_keywords[n:] if score > 0.45]
    

def extract_keywords_keybert(text, model):
    kw_model = keybert.KeyBERT(model=model)
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2))

