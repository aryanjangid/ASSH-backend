from flask import Flask, render_template, url_for, request,Response,jsonify     
from flask_pymongo import pymongo

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

app = Flask(__name__)

# mongo connection
CONNECTION_STRING = "mongodb+srv://admin:admin123@cluster0.csvwlpa.mongodb.net/?retryWrites=true&w=majority"

client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('flask_mongodb_atlas')
user_collection = pymongo.collection.Collection(db, 'user_collection')



# Load the TF-IDF vocabulary specific to the category
with open(r"toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open(r"toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)

# Render the HTML file for the home page
@app.route("/data",methods=['GET'])
def home():
    holder = list()
    currentCollection =db.db.collection
    for i in currentCollection.find():
        holder.append({'sentence':i['sentence']})
    return jsonify(holder)


@app.route("/", methods=['POST'])
def predict():
    # Take a string input from user
    currentCollection =db.db.collection
    user_input = request.json['sentence']
    currentCollection.insert_one({'sentence':user_input})
    
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1]

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1]

    out_tox = round(pred_tox[0], 2)
    out_sev = round(pred_sev[0], 2)
    out_obs = round(pred_obs[0], 2)
    out_ins = round(pred_ins[0], 2)
    out_thr = round(pred_thr[0], 2)
    out_ide = round(pred_ide[0], 2)

    # print(out_tox)
    response_body={
        "toxic":out_tox,
        "severe_toxic":out_sev,
        "obscene":out_obs,
        "insult":out_ins,
        "threat":out_thr,
        "identity_hate":out_ide,
    }
    return response_body

     
# Server reloads itself if code changes so no need to keep restarting:
app.run(debug=True)