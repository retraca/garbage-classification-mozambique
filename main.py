from data import Data
from questions import Questions
from images import Images
from model import Model

#smaller dataset (120 samples) with stefan groundtruth
""" stefan_data_path = 'stefanGroundtruth120.csv'
data_stefan = Data(stefan_data_path).get_all_data() """


#bigger dataset with field groundtruth
all_picture_data_path = 'versions/3000relevantquestions.csv'
data_all = Data(all_picture_data_path).rescale_grountruth()
#all_picture_fields=['image1', 'image2', 'image3', 'image4', 'image5', 'loc_dirty']
#data_simple = data_all.get_data_fields(all_picture_fields)
data_simple = data_all.get_all_data()
print(data_simple)

images_path = 'D:/Projects/2022/GarbageDetection/images'

model_path_vqa = "dandelin/vilt-b32-finetuned-vqa"
positives = ["Is there trash?", "Is there garbage?", "Is there waste?", "Is there junk?", "Is there residue?", "Is there rubbish?", "Is there a lot of trash?", "Is there a lot of garbage?", "Is there a lot of waste?", "Is there a lot of junk?", "Is there a lot of residue?", "Is there a lot of rubbish?", "Is there any type of trash?", "Is there any type of garbage?", "Is there any type of waste?", "Is there any type of junk?", "Is there any type of residue?", "Is there any type of rubbish?", "Are there any bottles on the ground?", "Are there any bags on the ground?", "Is there any plastic on the ground?", "Is there any paper on the ground?", "Are thre any leaves on the ground?", "Are there plants on the ground?", "Are there any branches on the ground?", "Is there any metal on the ground?", "Are there any packages on the ground?", "Is there any stuff on the ground?", "Are there any times on the ground?", "Are there any objects on the ground?", "Is there anything in the water?", "Are there small items on the ground?"]
negatives = ["Is there a lot of water?", "Is there water?", "Is there grass?", "Is there sand?", "Are there buildings?", "Are there structures?", "Are there bricks?", "Are there walls?", "Is the ground empty?", "Is the ground sand?", "Are there stones?", "Is there concrete", "Is the ground clean?", "Are there shadows on the ground?", "Are there dark and light areas on the ground?", "Are there houses?", "Does the ground have different colors?", "Are there people?", "Are there clothes?", "Is there laundry?", "Are there animals?", "Are there birds?", "Are there trees?"]

model_path_nlvr = "dandelin/vilt-b32-finetuned-nlvr2"
reasoning_classes= ['nlvr_class1', 'nlvr_class2', 'nlvr_class3', 'nlvr_class4', 'nlvr_class5']
reasoning_questions=['The left image has a lot more trash than the right image', 'The left image has a lot more waste than the right image', 'The left image shows a lot more garbage than the right image']

images = Images(images_path)
model = Model(model_path_vqa, model_path_nlvr)
questions = Questions(positives, negatives)

#TODO: get best questions from data
""" 
best_positives, best_negatives = questions.get_best_questions(data, images, model, 5)
print(best_positives, best_negatives) 
"""
""" all_questions= positives + negatives
best_indexes= [3,6,7,8,11,15,25,33,34,35,36,37,39,41,42,47,54]

best_positives=[]
best_negatives=[]
idx=-1
for qst in all_questions:
    idx+=1
    if idx in best_indexes:
        #print(idx, qst)
        if idx < 32:
            best_positives.append(qst)
        else:
            best_negatives.append(qst) """ 

#table_scores= model.get_relevant_questions_data(data_simple, images, best_positives, best_negatives) 
table_scores= model.get_relevant_questions_data(data_simple, images, positives, negatives, reasoning_classes, reasoning_questions) 


#print(table_scores) 
table_scores.to_csv(r'D:\Projects\2022\GarbageDetection\CLEAN\main.csv', index = True, header=True)
