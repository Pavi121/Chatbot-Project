import random
from flask import Flask, redirect, render_template, request, url_for
import numpy as np
from tensorflow.keras.models import load_model
import json
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
from googletrans import Translator
import mysql.connector

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
# USERNAME = 'admin'
# PASSWORD = '1234'
mydb = mysql.connector.connect(host="localhost", user="root", password="", database="bot")
mycursor = mydb.cursor()

# Load the trained model and other necessary files
model = load_model('bot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open('bot.json', 'r') as file:
    data = json.load(file)

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()
translator = Translator()

def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def translate_to_tamil(text):
    try:
        translated_text = translator.translate(text, dest='ta').text
    except Exception as e:
        print("Translation failed:", e)
        translated_text = text
    return translated_text

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/loginpost', methods=['POST', 'GET'])
def userloginpost():
    global data1
    if request.method == 'POST':
        data1 = request.form.get('uname')
        data2 = request.form.get('password')
        
        print("Username:", data1)  # Debug statement
        print("Password:", data2)  # Debug statement

        if data2 is None:
            return render_template('login.html', msg='Password not provided')

        sql = "SELECT * FROM `users` WHERE `uname` = %s AND `password` = %s"
        val = (data1, data2)

        try:
            mycursor.execute(sql, val)
            account = mycursor.fetchone()  # Fetch one row

            if account:
                # Consume remaining results
                mycursor.fetchall()
                mydb.commit()
                return redirect(url_for('dash'))
            else:
                return render_template('login.html', msg='Invalid username or password')
        except mysql.connector.Error as err:
            print("Error:", err)  # Debug statement
            return render_template('login.html', msg='An error occurred. Please try again.')



@app.route('/NewUser')
def newuser():
    return render_template('NewUser2.html')

@app.route('/reg', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        uname = request.form.get('uname')
        email = request.form.get('email')
        phone = request.form.get('phone')
        age = request.form.get('age')
        password = request.form.get('psw')
        gender = request.form.get('gender')
        sql = "INSERT INTO users (name, uname, email , phone, age, password, gender) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (name, uname, email, phone, age, password, gender)
        mycursor.execute(sql, val)
        mydb.commit()
        return render_template('login.html')
    else:
        return render_template('NewUser2.html')

@app.route('/dash')
def dash():
    return render_template('dash.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    message = request.form["message"]
    ints = predict_class(message, model)
    response = ''
    for i in data["intents"]:
        if i["tag"] == ints[0]["intent"]:
            response = random.choice(i["responses"])
            break

    translated_response = translate_to_tamil(response)
    return render_template('result.html', response=translated_response)
@app.route('/gov')
def gov():
    return render_template('grievance.html')
@app.route('/gov1')
def gov1():
    return render_template('pressrelease.html')
@app.route('/gov2')
def gov2():
    return render_template('deptname.html')
@app.route('/gov3')
def gov3():
    return render_template('Sitemaps.html')
@app.route('/gov4')
def gov4():
    return render_template('aboutus.html')
@app.route('/gov5')
def gov5():
    return render_template('contact.html')
@app.route('/gov6')
def gov6():
    return render_template('contactus.html')
@app.route('/gov7')
def gov7():
    return render_template('holiday.html')
@app.route('/gov8')
def gov8():
    return render_template('announcements.html')
@app.route('/gov9')
def gov9():
    return render_template('dept.html')
@app.route('/gov10')
def gov10():
    return render_template('contact_directory.html')
@app.route('/gov11')
def gov11():
    return render_template('ta.html')

if __name__ == "__main__":
    app.run(debug=True)
