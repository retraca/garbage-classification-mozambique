import pandas as pd

class Data:
    
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as f:
            self.data = pd.read_csv(f)
    
    def get_all_data(self):
        return self.data
    
    def get_data_fields(self, fields):
        return self.data[fields]
    
    def rescale_grountruth(self):
        #self.data['loc_dirty'] = self.data['loc_dirty'].apply(lambda x: str(x/2).split('.')[0])
        return self
    
    