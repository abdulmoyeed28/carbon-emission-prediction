from flask import *
import joblib
import numpy as np
from datetime import datetime
import pickle
from views import preprocess

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/user')
def user():
    return render_template("user.html")


@user_bp.route('/user_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("index.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)

@user_bp.route('/predict1')
def predict1():
    return render_template("index.html")




@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")