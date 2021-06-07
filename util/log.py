"""
log.py
"""

class log:
    def __init__(self,f,name = "", PRINT = True, retrain=False):
        text = ""
        if type(name) == list:
            text = "{}".format(name[0])
            for x in name[1:]:
                text += ",{}".format(x)
        elif type(text) == str:
            text = name
        self.FNAME = f
        if retrain:
            F = open(self.FNAME,"a")
        else: 
            F = open(self.FNAME,"w+")
        if len(text) != 0:
            F.write(text + "\n")
        F.close()
        if PRINT:
            print(text)
    def log(self,data,PRINT = True):
        text = ""
        if type(data) == list:
            text = "{}".format(data[0])
            for x in data[1:]:
                text += ",{}".format(x) 
        elif type(data) == str:
            text = data
        if PRINT:
            print(text)
        F = open(self.FNAME,"a")
        F.write(str(text) + "\n")
        F.close()