import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Load credentials
if not firebase_admin._apps:
    if os.environ.get("FIREBASE_CREDENTIALS"):
        # Running on Render (env variable)
        cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS"])
        cred = credentials.Certificate(cred_dict)
    else:
        # Running locally (file)
        cred = credentials.Certificate("firebase-key.json")

    firebase_admin.initialize_app(cred)

db = firestore.client()
