

class KeyValueAnalyzer:
    def process_batch(self):
        pass



"""
with fmrai():
    model = instrument_model(...)

    kv = KeyValueAnalyzer()
    
    for batch in ...:
        with kv.process_batch():
            model(**batch)

"""