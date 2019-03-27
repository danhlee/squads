from flask import Flask, jsonify, request, Response, json
from classifier import get_model, get_prediction, TREE, BAYES
from flask_sqlalchemy import SQLAlchemy
from models import db

app = Flask(__name__)
POSTGRES = {
    'user': 'postgres',
    'pw': 'password',
    'db': 'my_database',
    'host': 'localhost',
    'port': '5432',
}
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://DB_USER:PASSWORD@HOST/DATABASE'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
db.init_app(app)

@app.route('/')
def index():
    return 'Welcome to SQUADS!'

@app.route('/predict', methods=['POST'])
def predict():
    json_roster = request.json['roster']
    
    # Verify incoming JSON roster has necessary values (each position)
    print('roster =', json_roster)
    if validateRoster(json_roster):

        # creates model according to string arg
        model = get_model(TREE)
        # get predicted class using model and roster input from user
        json_prediction = get_prediction(model, json_roster)

        # create response object and return to user
        response = Response(response=json.dumps(json_prediction), status=202, mimetype='application/json')
        return response
    else:
        return Response(response="Invalid Request!", status=400, mimetype='text/plain')

def validateRoster(roster):
    if ("b_bot" in roster and "b_jung" in roster and "b_mid" in roster and "b_sup" in roster and "b_top" in roster and"r_bot" in roster and"r_jung" in roster and "r_mid" in roster and "r_sup" in roster and "r_top" in roster):
        return True
    else:
        return False

if __name__ == '__main__':
    app.run(debug=True)