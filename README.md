# Spam Email Classifier

An AI-powered text classification tool designed to filter out spam emails with high precision. This project utilizes Natural Language Processing (NLP) and Machine Learning to distinguish between "Ham" (legitimate) and "Spam" messages.

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
```text
├── app.py                # Streamlit web application
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # Saved TF-IDF vectorizer
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
