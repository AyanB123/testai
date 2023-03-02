import googlesearch
import requests
import bs4
import torch
import sklearn 
import spacy
import openai
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn, optim

# initialize OpenAI API
openai.api_key = "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"

# perform sentiment analysis
def sentiment_analysis(text):
    # use spacy library to analyze sentiment
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentiment = doc.sentiment
    return sentiment

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
        engine="text-davinci-003",
        prompt=X,
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.5,
    )[0].text
    return answer

# define function to train neural network
def train_neural_network():
    # use past question-answer pairs to train neural network
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X = torch.tensor([]) # tensor of features
    y = torch.tensor([]) # tensor of labels
    for epoch in range(10):
        running_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            output = model(X[i])
            loss = criterion(output, y[i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch:", epoch+1, "Loss:", running_loss/len(X))
    return model

# define function to select best answer

def select_best_answer(answers):
    # use neural network to select best answer from multiple runs of GPT API
    model = train_neural_network()
    scores = []
    for answer in answers:
        X = torch.tensor([sentiment_analysis(answer),
                          select_most_relevant_info(search_for_info(answer))])
        score = model(X)
        scores.append(score.item())
    best_answer = answers[np.argmax(scores)]
    return best_answer
# define function to combine question and relevant information
def combine_question_and_info(question, info):
    # combine question and information
    new_question = question + " " + info
    return new_question


def main():
    # initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # fit vectorizer on corpus
    corpus = []
    for file_name in os.listdir("corpus"):
        with open(os.path.join("corpus", file_name), 'r') as f:
            text = f.read()
            corpus.append(text)
    vectorizer.fit(corpus)

    # ask for user input
    question = input("What is your question? ")

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