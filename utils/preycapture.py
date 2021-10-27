class PreyCapture(BaseInput):
    def __init__(self, config, recording_name, recording_path, has_opto=False):
        super.__init__(self, config, recording_name, recording_path)

        self.has_opto = has_opto

    def opto_analysis(self)

    def process(self)