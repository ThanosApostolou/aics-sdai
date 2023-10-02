class FirebaseUser:
    def __init__(self, firebase_user_id: str, email: str, name: str):
        self.firebase_user_id = firebase_user_id
        self.email = email
        self.name = name


    def __repr__(self):
        return str(self.__dict__)