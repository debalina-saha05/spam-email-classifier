# Spam Email Classifier

An AI-powered text classification tool designed to filter out spam emails with high precision. This project utilizes Natural Language Processing (NLP) and Machine Learning to distinguish between "Ham" (legitimate) and "Spam" messages.

##  Live Demo
[**Click here to view the Live Web App**](https://spam-email-classifier-project1.streamlit.app/)

## Key Features
* **High Accuracy:** Achieved **96.68%** accuracy using the Multinomial Naive Bayes algorithm.
* **Real-time Prediction:** User-friendly interface built with Streamlit for instant classification.
* **NLP Pipeline:** Includes text preprocessing, tokenization, and TF-IDF vectorization.

## Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Algorithm** | Multinomial Naive Bayes |
| **Accuracy** | 96.68% |
| **Vectorization** | TF-IDF / Bag of Words |

## Tech Stack
* **Language:** Python
* **ML Library:** Scikit-Learn
* **Data Handling:** Pandas, NumPy
* **UI/UX:** Streamlit

## Project Structure
* `app.py`: The core Streamlit application script for the web interface.
* `train.py`: The Machine Learning pipeline (Data cleaning, Preprocessing, and Model training).
* `model.pkl`: The trained and serialized Naive Bayes model.
* `vectorizer.pkl`: The saved TF-IDF vectorizer for processing new inputs.
* `requirements.txt`: List of Python dependencies (scikit-learn, streamlit, etc.).
```
