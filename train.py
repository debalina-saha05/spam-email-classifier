import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load the new email dataset
# This dataset usually has columns 'label' and 'text'
df = pd.read_csv('emails.csv')

# Ensure we have the right columns and no empty data
df = df[['text', 'label']].dropna()
df['text'] = df['text'].astype(str)

# 2. Vectorize (Convert email text to numbers)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label']

# 3. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Save the "Brain" files
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

print(f"Success! Trained on {len(df)} Emails.")
print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")