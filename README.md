# Carbon Emission Prediction

A machine learning-based web application built using **Flask** that predicts carbon emissions from vehicle and environmental data.  
The project also includes algorithm performance analysis and data visualizations to compare multiple ML models.

---

## **Features**
- Predict carbon emissions using a trained ML model.
- Compare multiple algorithms (Lasso, XGBoost, AdaBoost, etc.).
- Data analysis and visualizations using built-in datasets.
- Separate user and admin panels with a simple web interface.

---

## **Tech Stack**
- **Backend:** Python, Flask  
- **Frontend:** HTML, CSS (Bootstrap)  
- **Machine Learning:** scikit-learn, XGBoost, mlxtend  
- **Visualization:** Matplotlib, Seaborn  

---

## **Project Structure**

CarbonEmissionPrediction/
│
├── app.py # Flask app entry point
├── requirements.txt # Python dependencies
├── views/ # Blueprint routes (user & admin)
├── data/ # ML model training and analysis scripts
├── templates/ # HTML templates
├── static/ # CSS, images, and visualization outputs
├── finalized_model.pkl # Trained model (optional)
├── final_co2.csv # Sample dataset (optional)
└── CO2 Emissions_Canada.csv # Sample dataset (optional)


## **Installation & Setup**

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/carbon-emission-prediction.git
   cd carbon-emission-prediction

2. Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3.Install dependencies


pip install -r requirements.txt

4. Run the Flask app

python app.py

5. Open in browser

http://127.0.0.1:5000/


Requirements
Here are the main dependencies used:

Copy
Edit
numpy~=1.24.4
pandas~=2.0.3
seaborn~=0.13.2
matplotlib~=3.7.5
scikit-learn~=1.3.2
flask~=3.0.3
statsmodels~=0.14.1
scipy~=1.10.1
mlxtend~=0.23.4
xgboost~=2.1.4
joblib~=1.4.2