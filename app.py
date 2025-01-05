import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import requests
from bs4 import BeautifulSoup


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
import matplotlib.pyplot as plt

from helpers.helper import get_img, get_processed, get_reviews, download_stopwords
from flask_session import Session


app = Flask(__name__)
app.secret_key = "123"
app.config["SESSION_TYPE"] = "filesystem"  # Use filesystem for session storage
app.config["SESSION_PERMANENT"] = False  # Sessions are not permanent by default
app.config["SESSION_FILE_DIR"] = "./flask_sessions"

Session(app)

df = None
lstm_model = None
suggestions = pd.DataFrame()


def preload():
    global df
    global lstm_model
    global suggestions
    df = pd.read_csv(r"./static/db/movie.csv")
    lstm_model = load_model(r"./static/db/lstm123.keras")
    download_stopwords()


def getimage():
    urldb = []
    imagedb = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36"
    }

    for t in df["tconst"]:
        url = f"https://www.imdb.com/title/{t}"
        urldb.append(url)

    for url in urldb:
        result = requests.get(url, headers=headers)
        soup = BeautifulSoup(result.text, "html.parser")
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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        query = request.args.get("query", "")
        if query:
            session["search_text"] = query
            return redirect(url_for("search"))
    return render_template(
        "index.html",
    )


@app.route("/search", methods=["GET", "POST"])
def search():
    global suggestions
    if request.method == "POST":
        # Get the selected index from the form
        selected_index = request.form.get("selected_index")
        # print(selected_index)
        session["selected_index"] = selected_index
        if selected_index:
            return redirect(url_for("reviews"))
            # qs = session.get("search_text", None)
            # print("QUERY", qs)
            # if qs:
            #     # suggestions = session.get("suggestions")
            #     print("YES", suggestions)
            #     if not suggestions.empty:
            #         print("Yes")
            #         sugg = suggestions[
            #             suggestions["primaryTitle"].str.contains(
            #                 qs, case=False, na=False
            #             )
            #         ]
            #         selected_movie = sugg.iloc[int(selected_index)]
            #         tconst = selected_movie["tconst"]
            #         session["tconst"] = tconst
            #         print(tconst)
            #         # return f"Hello {tconst}"
            #         return redirect(url_for("reviews"))
            # return render_template("review.html", tconst=tconst)

    query = session.get("search_text", [])
    if query:
        # global suggestions
        suggestions = df[df["primaryTitle"].str.contains(query, case=False, na=False)]
        # print(suggestions)
        # imgdata = get_img(suggestions)
        imgdata = [None] * len(suggestions)
        suggestions["imgurl"] = imgdata
        session["suggestions"] = suggestions
        dft_list = suggestions.to_dict(orient="records")
    return render_template(
        "suggestions.html",
        dft_list=dft_list,
        suggestions=suggestions,
        svg_file="images/no.svg",
    )


@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    qs = session.get("search_text", None)
    selected_index = session.get("selected_index")
    print("SEL", selected_index)
    if qs:
        suggestions = session.get("suggestions")
        if suggestions is not None and not suggestions.empty:
            sugg = suggestions[
                suggestions["primaryTitle"].str.contains(qs, case=False, na=False)
            ]
            if selected_index is not None:
                selected_movie = sugg.iloc[int(selected_index)]
                tconst = selected_movie["tconst"]
                print("TConst", tconst)
                if tconst:
                    reviews = get_reviews(tconst)
                    # reviews.to_csv("test.csv")
                    if reviews is not None and not reviews.empty:
                        unseen_padded = get_processed(reviews["user_review"])
                        print(unseen_padded)

                        unseen_sentiments = lstm_model.predict(unseen_padded)
                        # print(unseen_sentiments)
                        reviews["predicted_score"] = np.round(unseen_sentiments * 10, 1)

                        # print(reviews)
                        # print(reviews.head())
                        reviews_list = reviews.to_dict(orient="records")
                        return render_template("review.html", reviews=reviews_list)
                    else:
                        return render_template("noreview.html")
    return "Failed"
    # return f"{reviews['user_review'][0]}"


if __name__ == "__main__":
    preload()
    app.run(debug=True)
