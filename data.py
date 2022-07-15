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
    
    def split_train_and_test(self):
        x = self.data.drop(['loc_dirty'], axis=1).drop(['scaled_loc_dirty'], axis=1)
        print(x)
        y= self.data['scaled_loc_dirty']
        x_train = x.sample(frac=0.8, random_state=1)
        x_test = x.drop(x_train.index)
        y_train = y.loc[x_train.index]
        y_test= y.loc[x_test.index]
        print('xtrain', x_train)
        print('ytrain', y_train)
        print('xtest', x_test)
        print('ytest', y_test)
        """ train = self.data[:int(len(self.data)*0.8)]
        test = self.data[int(len(self.data)*0.8):]
        x_train = train[['image1', 'image2', 'image3', 'image4', 'image5']]
        X_test = test[['image1', 'image2', 'image3', 'image4', 'image5']] """
        return x_train, x_test, y_train, y_test
    
    