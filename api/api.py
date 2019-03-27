from flask import Flask, jsonify, request

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]


@app.route('/')
def index():
    return "Hello, World!"

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/predict', methods=['POST'])
def predict():
    roster = request.json['roster']

    print('roster = ', roster)
    if validateRoster(roster):
        return jsonify(roster)
    else:
        return 'INVALID ROSTER'

def validateRoster(roster):
    if ("b_bot" in roster and "b_jung" in roster and "b_mid" in roster and "b_sup" in roster and "b_top" in roster and"r_bot" in roster and"r_jung" in roster and "r_mid" in roster and "r_sup" in roster and "r_top" in roster):
        return True
    else:
        return False

if __name__ == '__main__':
    app.run(debug=True)