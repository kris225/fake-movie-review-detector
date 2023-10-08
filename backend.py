import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_url_path='/static', static_folder='static')

nltk.download('movie_reviews')
nltk.download('punkt')

# Load the movie reviews dataset
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

random.shuffle(reviews)

# Prepare the data
documents = [" ".join(review) for review, category in reviews]
labels = [category for review, category in reviews]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Create TF-IDF vectors
vectorizer = CountVectorizer(max_features=5000)
X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Transform test data and make predictions
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)



# ... (Data preprocessing and model training code)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/detector')
def detector():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_review = request.form['review']

    # Preprocess the input review
    input_review = " ".join(nltk.word_tokenize(input_review))

    # Transform the input using the same vectorizer and TF-IDF transformer
    input_review_counts = vectorizer.transform([input_review])
    input_review_tfidf = tfidf_transformer.transform(input_review_counts)

    # Make predictions
    prediction = clf.predict(input_review_tfidf)

    if prediction == 'pos':
        result = "The review was made by a Real Person."
    else:
        result = "The review is a Fake."

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
