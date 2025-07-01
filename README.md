📩 SMS Spam Detector
A simple web app built using Streamlit that classifies SMS messages as \*\*Spam\*\* or \*\*Ham\*\* using a trained Naive Bayes model.

💡 How it Works
\- The model was trained on the `spam.csv` dataset using CountVectorizer and Multinomial Naive Bayes.

\- Once trained, the model and vectorizer are saved using `joblib`.

\- This app loads those files and classifies user input in real time.



\## 🚀 Run Locally



```bash

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

streamlit run app.py



