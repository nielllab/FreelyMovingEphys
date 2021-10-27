class Error(Exception):
    pass

class EyeTrackingError(Error):
    def __init__(self, message='Unknown error in eye tracking.'):
        self.message = message

class UserInputError(Error):
    def __init__(self, message='Unknown error in user input or settings.'):
        self.message = message