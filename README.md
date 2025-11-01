# Insurance-Premium-Category-Predictor

A web application to predict insurance premium categories based on user details using a trained ML model. The app consists of:

FastAPI backend: Serves the prediction API.

Streamlit frontend: Interactive web interface for users to input their details.

ML Model: Pre-trained model (model.pkl) for premium prediction.

Features

Calculates BMI, lifestyle risk, age group, and city tier automatically.

Predicts insurance premium category: Low / Medium / High.

Interactive frontend using Streamlit.
Running the Application
1️⃣ Start FastAPI backend
uvicorn app:app --reload


API will be available at: http://127.0.0.1:8000/predict

You can also view automatic docs at: http://127.0.0.1:8000/docs

2️⃣ Start Streamlit frontend
streamlit run streamlit_app.py


Streamlit app will open in your browser.

Enter your details and click Predict Premium Category.


