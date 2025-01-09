import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import nltk
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

# Setting header as to simulate as a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36"
}


# Get all image links from the website
def get_img(df):
    urldb = []
    imagedb = []
    for t in df["tconst"]:
        url = f"https://www.imdb.com/title/{t}"
        urldb.append(url)

    for url in urldb:
        result = requests.get(url, headers=headers)
        soup = bs(result.text, "html.parser")
        div = soup.find("div", class_="ipc-media")
        img = div.find("img", class_="ipc-image")
        if img:
            img_src = img.get("src")
        else:
            img_src = None
        imagedb.append(img_src)

    return imagedb


# Scapes all user scores and reviews
def get_reviews(tconst):
    review_score = []
    review_review = []
    ## The constucted link
    url = f"https://www.imdb.com/title/{tconst}/reviews"
    page = requests.get(url, headers=headers)
    soup = bs(page.content, "html.parser")
    reviews = soup.find_all("article", class_="user-review-item")
    if reviews:
        for review in reviews:
            rating = review.find("span", class_="ipc-rating-star")
            reviewtext = review.find(attrs={"data-testid": "review-overflow"})
            print(rating)
            if reviewtext is not None:
                if rating is None:
                    rating = "NA"
                    review_score.append(rating)
                else:
                    review_score.append(rating.get_text(strip=True))
                review_review.append(reviewtext.get_text(strip=True))

        review_data = pd.DataFrame(
            {"user_review": review_review, "user_score": review_score}
        )
        return review_data
    else:
        return None


# Regex for HTML Tags
TAG_RE = re.compile(r"<[^>]+>")


# Remove the html tags
def remove_tags(text):
    return TAG_RE.sub("", text)


# Preprocessing the Text
def preprocess_text(sen):
    sentence = sen.lower()

    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Single character removal
    sentence = re.sub(
        r"\s+[a-zA-Z]\s+", " ", sentence
    )  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(
        r"\s+", " ", sentence
    )  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords.words("english")) + r")\b\s*")
    sentence = pattern.sub("", sentence)

    return sentence


# Preprocessing on user reviews
def get_processed(unseen_reviews):
    unseen_processed = []
    for review in unseen_reviews:
        review = preprocess_text(review)
        unseen_processed.append(review)

    word_tokenizer = Tokenizer()
    maxlen = 100

    word_tokenizer.fit_on_texts(unseen_processed)
    unseen_tokenized = word_tokenizer.texts_to_sequences(unseen_processed)

    word_tokenizer.fit_on_texts(unseen_processed)

    unseen_padded = pad_sequences(unseen_tokenized, padding="post", maxlen=maxlen)

    return unseen_padded


# Download Stopwords
def download_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
        print("Stopwords are already downloaded.")
    except LookupError:
        print("Stopwords not found, downloading...")
        nltk.download("stopwords")

    from nltk.corpus import stopwords

    return stopwords.words("english")
