import datetime

def preprocess():
    # Set the expiration date
    expiration_date = datetime.date(2025, 6, 30)

    # Get today's date
    today = datetime.date.today()
    msg=""
    # Check if today's date is past the expiration date
    if today > expiration_date:
        msg="invalid"
    else:
        msg="valid"

    return msg
