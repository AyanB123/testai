import googlesearch
import requests
import bs4
import tensorflow as tf
import sklearn 
import spacy
import openai
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow  import Sequential





# initialize OpenAI API
openai.api_key = "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"

# perform sentiment analysis
def sentiment_analysis(text):
    # use spacy library to analyze sentiment
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentiment = doc.sentiment
    return sentiment

# read academic questions from text file
def read_questions_from_file(file_path):
    with open(file_path, 'r') as f:
        questions = f.read().splitlines()
    return questions

# define function to search for relevant information
def search_for_info(question):
    # use googlesearch library to search question on Google
    search_results = googlesearch.search(question, num_results=10)
    # retrieve HTML content of the first search result
    response = requests.get(next(search_results))
    soup = bs4.BeautifulSoup(response.content, 'html.parser')
    # use spacy library to extract relevant information from the text of the page
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(soup.get_text())
    relevant_info = ""
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "DATE", "GPE"]:
            relevant_info += ent.text + " "
    return relevant_info

# define function to select most relevant information
def select_most_relevant_info(info):
    # use TF-IDF to select most relevant information
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([info])
    features = vectorizer.get_feature_names()
    top_indices = np.argsort(X.toarray())[0][-10:]  # get top 10 features
    most_relevant_info = " ".join([features[i] for i in top_indices])
    return most_relevant_info

# define function to generate answer
def generate_answer(X):
    # use GPT API to generate answer
    answer = openai.Completion.create(
        engine="davinci",
        prompt=X,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )[0].text
    return answer

# define function to train neural network
def train_neural_network():
    # use past question-answer pairs to train neural network
    model = Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    X = [] # list of features
    y = [] # list of labels
    model.fit(X, y, epochs=10)
    return model

# define function to select best answer
def select_best_answer(answers):
    # use neural network to select best answer from multiple runs of GPT API
    model = train_neural_network()
    scores = []
    for answer in answers:
        X = [] # list of features
        score = model.predict(X)
        scores.append(score)
    best_index = scores.index(max(scores))
    best_answer = answers[best_index]
    return best_answer

# define function to combine question and relevant information
def combine_question_and_info(question, info):
    # combine question and information
    new_question = question + " " + info
    return new_question



def main():
    # read academic questions from text file
    questions = read_questions_from_file("questions.txt")

    # initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # fit vectorizer on corpus
    corpus = []
    for file_name in os.listdir("corpus"):
        with open(os.path.join("corpus", file_name), 'r') as f:
            text = f.read()
            corpus.append(text)
    vectorizer.fit(corpus)

    # iterate over questions
    for question in questions:
        # search for relevant information
        info = search_for_info(question)
        # select most relevant information
        relevant_info = select_most_relevant_info(info)
        # combine question and relevant information
        new_question = combine_question_and_info(question, relevant_info)

        # generate feature vector using TF-IDF
        X = vectorizer.transform([new_question])

        # generate answer
        answers = []
        for i in range(3):  # run GPT API multiple times
            answer = generate_answer(X)
            answers.append(answer)

        # select best answer
        best_answer = select_best_answer(answers)

        # print best answer
        print("Q:", question)
        print("A:", best_answer)

if __name__ == "__main__":
    main()