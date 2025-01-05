import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import nltk
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36"
}


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
        # print(div)
        img = div.find("img", class_="ipc-image")
        if img:
            img_src = img.get("src")
        else:
            img_src = None
        imagedb.append(img_src)
        # print(img_src)

    return imagedb


def get_reviews(tconst):
    review_score = []
    review_review = []
    url = f"https://www.imdb.com/title/{tconst}/reviews"
    # print(url)
    page = requests.get(url, headers=headers)
    soup = bs(page.content, "html.parser")
    # print(soup.prettify())
    reviews = soup.find_all("article", class_="sc-d99cd751-1 kzUfxa user-review-item")
    if reviews:
        for review in reviews:
            rating = review.find("span", class_="ipc-rating-star--rating")
            # reviewtext = review.find(attrs={"data-testid": "review_overflow"})
            reviewtext = review.find(attrs={"data-testid": "review-overflow"})
            # if reviewtext is not None:
            #     print(reviewtext.get_text(strip=True))
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


TAG_RE = re.compile(r"<[^>]+>")


def remove_tags(text):
    return TAG_RE.sub("", text)


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


def get_processed(unseen_reviews):
    unseen_processed = []
    for review in unseen_reviews:
        review = preprocess_text(review)
        unseen_processed.append(review)
        # print(unseen_processed)
    # df = pd.DataFrame(unseen_processed)
    # df.to_csv("unseen.csv")

    # print(len(unseen_processed))
    word_tokenizer = Tokenizer()
    maxlen = 100

    word_tokenizer.fit_on_texts(unseen_processed)
    unseen_tokenized = word_tokenizer.texts_to_sequences(unseen_processed)

    word_tokenizer.fit_on_texts(unseen_processed)

    unseen_padded = pad_sequences(unseen_tokenized, padding="post", maxlen=maxlen)

    # df = pd.DataFrame(unseen_padded)
    # print(unseen_padded.shape)
    # print(unseen_padded)
    return unseen_padded


def download_stopwords():
    """
    This function checks if the NLTK stopwords dataset is available.
    If not, it downloads it.
    """
    try:
        # Check if stopwords are already downloaded
        nltk.data.find("corpora/stopwords")
        print("Stopwords are already downloaded.")
    except LookupError:
        # If stopwords are not found, download them
        print("Stopwords not found, downloading...")
        nltk.download("stopwords")

    # Return the stopwords list
    from nltk.corpus import stopwords

    return stopwords.words("english")
