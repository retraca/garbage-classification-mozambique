import os

class Images:
    def __init__(self, images_path):
        self.images_path = images_path
        
    def get_viz_images_info(self, data, firstimage):
        info= []
        folders = os.listdir(self.images_path)
        for index, row in data.iterrows():
            if row['image1'] == firstimage:
                #TODO loc_dirty in bigger dataset is stefan in smaller dataset, just change the name in csv
                fields = [row.image1, row.image2, row.image3,row.image4, row.image5, row.loc_dirty]
                idx_field = 0
                for field in fields:
                    if idx_field > 4:
                        info.append(field)
                    for folder in folders: 
                        files = os.listdir(f'{self.images_path}/{folder}')
                        if field in files:
                            info.append(f'{self.images_path}/{folder}/{field}')
                            idx_field += 1
        return info
        

