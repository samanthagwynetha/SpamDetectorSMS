📩 SMS Spam Detector
A simple web app built using Streamlit that classifies SMS messages as Spam or Ham using a trained Naive Bayes model.

📸: Screenshots

Home:
![image](https://github.com/user-attachments/assets/3050ee39-ef16-442e-8cea-d3e37f2248a2)

Spam: 
![image](https://github.com/user-attachments/assets/296127c0-f938-46f8-8884-0a91e0d08f95)

Ham: 
![image](https://github.com/user-attachments/assets/2f2071da-f09c-4d4d-a7f6-7c543e604a5f)



💡 How it Works

\- The model was trained on the `spam.csv` dataset using CountVectorizer and Multinomial Naive Bayes.

\- Once trained, the model and vectorizer are saved using `joblib`.

\- This app loads those files and classifies user input in real time.



🚀 Run Locally

```bash

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py



