import firebase_admin

from firebase_admin import credentials

def create_firebase_app() -> firebase_admin.App:
    cred = credentials.Certificate("./firebaseServiceAccountKey.json")

    firebaseApp = firebase_admin.initialize_app(cred)
    return firebaseApp
