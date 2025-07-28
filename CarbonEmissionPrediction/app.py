from flask import *
import pickle
import pandas as pd
import views.adminbp, views.userbp
from views import preprocess
import numpy as np
app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
app.secret_key = "abc"
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST' and preprocess()=="valid":

        Engine_Size = float(request.form['Engine_Size'])
        Cylinders = float(request.form['Cylinders'])
        Fuel_Consumption_City = float(request.form['Fuel_Consumption_City'])
        Fuel_Consumption_Hwy = float(request.form['Fuel_Consumption_Hwy'])
        Fuel_Consumption_Comb = float(request.form['Fuel_Consumption_Comb'])
        Fuel_Consumption_Comb_mpg = float(request.form['Fuel_Consumption_Comb_mpg'])

        Make = request.form['Make']
        if (Make == 'Luxury'):
            Make_Type_Luxury = 1
            Make_Type_Premium = 0
            Make_Type_Sports = 0

        elif (Make == 'Premium'):
            Make_Type_Luxury = 0
            Make_Type_Premium = 1
            Make_Type_Sports = 0

        elif (Make == 'Sports'):
            Make_Type_Luxury = 0
            Make_Type_Premium = 0
            Make_Type_Sports = 1

        else:
            Make_Type_Luxury = 0
            Make_Type_Premium = 0
            Make_Type_Sports = 0

        Vehicle_Class = request.form['Vehicle_Class']
        if (Vehicle_Class == 'SUV'):
            Vehicle_Class_Type_SUV = 1
            Vehicle_Class_Type_Sedan = 0
            Vehicle_Class_Type_Truck = 0

        elif (Vehicle_Class == 'Sedan'):
            Vehicle_Class_Type_SUV = 0
            Vehicle_Class_Type_Sedan = 1
            Vehicle_Class_Type_Truck = 0

        elif (Vehicle_Class == 'Truck'):
            Vehicle_Class_Type_SUV = 0
            Vehicle_Class_Type_Sedan = 0
            Vehicle_Class_Type_Truck = 1

        else:
            Vehicle_Class_Type_SUV = 0
            Vehicle_Class_Type_Sedan = 0
            Vehicle_Class_Type_Truck = 0

        Transmission = request.form['Transmission']
        if (Transmission == 'A4'):
            Transmission_A4 = 1
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0
        elif (Transmission == 'A5'):
            Transmission_A4 = 0
            Transmission_A5 = 1
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'A6'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 1
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'A7'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 1
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'A8'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 1
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'A9'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 1
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AM5'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 1
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AM6'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 1
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AM7'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 1
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AM8'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 1
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AM9'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 1
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS4'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 1
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS5'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 1
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS6'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 1
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS7'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 1
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS8'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 1
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS9'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 1
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AS10'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 1
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AV'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 1
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AV6'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 1
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AV8'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 1
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'AV10'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 1
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'M5'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 1
            Transmission_M6 = 0
            Transmission_M7 = 0

        elif (Transmission == 'M6'):
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 1
            Transmission_M7 = 0

        else:
            Transmission_A4 = 0
            Transmission_A5 = 0
            Transmission_A6 = 0
            Transmission_A7 = 0
            Transmission_A8 = 0
            Transmission_A9 = 0
            Transmission_AM5 = 0
            Transmission_AM6 = 0
            Transmission_AM7 = 0
            Transmission_AM8 = 0
            Transmission_AM9 = 0
            Transmission_AS10 = 0
            Transmission_AS4 = 0
            Transmission_AS5 = 0
            Transmission_AS6 = 0
            Transmission_AS7 = 0
            Transmission_AS8 = 0
            Transmission_AS9 = 0
            Transmission_AV = 0
            Transmission_AV10 = 0
            Transmission_AV6 = 0
            Transmission_AV7 = 0
            Transmission_AV8 = 0
            Transmission_M5 = 0
            Transmission_M6 = 0
            Transmission_M7 = 1

        Fuel_Type = request.form['Fuel_Type']
        if (Fuel_Type == 'Type_E'):
            Fuel_Type_E = 1
            Fuel_Type_X = 0
            Fuel_Type_Z = 0

        elif (Fuel_Type == 'Type_X'):
            Fuel_Type_E = 0
            Fuel_Type_X = 1
            Fuel_Type_Z = 0

        elif (Fuel_Type == 'Type_Z'):
            Fuel_Type_E = 0
            Fuel_Type_X = 0
            Fuel_Type_Z = 1

        else:
            Fuel_Type_E = 0
            Fuel_Type_X = 0
            Fuel_Type_Z = 0

        def lr(Engine_Size, Cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy, Fuel_Consumption_Comb,
               Fuel_Consumption_Comb_mpg, Fuel_Type_E, Fuel_Type_X, Fuel_Type_Z, Transmission_A4, Transmission_A5,
               Transmission_A6, Transmission_A7, Transmission_A8, Transmission_A9, Transmission_AM5, Transmission_AM6,
               Transmission_AM7, Transmission_AM8, Transmission_AM9, Transmission_AS10, Transmission_AS4,
               Transmission_AS5, Transmission_AS6, Transmission_AS7, Transmission_AS8, Transmission_AS9,
               Transmission_AV, Transmission_AV10, Transmission_AV6, Transmission_AV7, Transmission_AV8,
               Transmission_M5, Transmission_M6, Transmission_M7, Make_Type_Luxury, Make_Type_Premium, Make_Type_Sports,
               Vehicle_Class_Type_SUV, Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck):
            c = pd.DataFrame(
                [Engine_Size, Cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy, Fuel_Consumption_Comb,
                 Fuel_Consumption_Comb_mpg, Fuel_Type_E, Fuel_Type_X, Fuel_Type_Z, Transmission_A4, Transmission_A5,
                 Transmission_A6, Transmission_A7, Transmission_A8, Transmission_A9, Transmission_AM5, Transmission_AM6,
                 Transmission_AM7, Transmission_AM8, Transmission_AM9, Transmission_AS10, Transmission_AS4,
                 Transmission_AS5, Transmission_AS6, Transmission_AS7, Transmission_AS8, Transmission_AS9,
                 Transmission_AV, Transmission_AV10, Transmission_AV6, Transmission_AV7, Transmission_AV8,
                 Transmission_M5, Transmission_M6, Transmission_M7, Make_Type_Luxury, Make_Type_Premium,
                 Make_Type_Sports, Vehicle_Class_Type_SUV, Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck]).T
            return model.predict(c)

        prediction = lr(Engine_Size, Cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy, Fuel_Consumption_Comb,
                Fuel_Consumption_Comb_mpg, Fuel_Type_E, Fuel_Type_X, Fuel_Type_Z, Transmission_A4, Transmission_A5,
                Transmission_A6, Transmission_A7, Transmission_A8, Transmission_A9, Transmission_AM5,
                Transmission_AM6, Transmission_AM7, Transmission_AM8, Transmission_AM9, Transmission_AS10,
                Transmission_AS4, Transmission_AS5, Transmission_AS6, Transmission_AS7, Transmission_AS8,
                Transmission_AS9, Transmission_AV, Transmission_AV10, Transmission_AV6, Transmission_AV7,
                Transmission_AV8, Transmission_M5, Transmission_M6, Transmission_M7, Make_Type_Luxury,
                Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV, Vehicle_Class_Type_Sedan,
                Vehicle_Class_Type_Truck)
        return render_template('index.html', prediction_text="Co2 Emissions by car is {} gm/km".format(np.round(prediction[0], 2)))
    else:
        return render_template('index.html', prediction_text="Some error occurred")


if __name__ == '__main__':
    app.run()
