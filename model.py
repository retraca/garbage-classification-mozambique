from transformers import ViltProcessor, ViltForQuestionAnswering, AutoTokenizer, AutoConfig
import pandas as pd
import cv2 as cv

class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(model_path,  output_hidden_states=True, output_attentions=True)  
        self.processor = ViltProcessor.from_pretrained(model_path)
        self.model = ViltForQuestionAnswering.from_pretrained(model_path, config=self.config) 
        
    def vqa(self, question, image):
        VQ_encoding = self.processor(image, question, return_tensors="pt")
        outputs = self.model(**VQ_encoding, return_dict = True)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]

    def get_binary_accuracy_question_viz(self, question, viz_images_info, answer):
        result=0
        for loc in viz_images_info[:5]:
            image = cv.imread(f'{loc}')
            model_answer = self.vqa(question, image)
            if model_answer == answer:
                result+=1    
        sum_results = result/5
        gt = viz_images_info[5]/5
        if sum_results>=(gt-0.2) and sum_results<=(gt+0.2):
            score=1
        else:
            score=0
        return score
    
    def get_absolut_score_question_viz(self, question, viz_images_info, answer):
        result=0
        for loc in viz_images_info[:5]:
            image = cv.imread(f'{loc}')
            model_answer = self.vqa(question, image)
            if model_answer == answer:
                result+=1    
        score = result/5
        return score
    
    def get_relevant_questions_data(self, data, images, best_positives, best_negatives):
        for idx, row in data.iterrows():
            if(idx>1000):
                break
            print(idx)
            image_info = images.get_viz_images_info(data, row.image1)
            for question in best_positives:
                #print(data)
                score= self.get_absolut_score_question_viz(question, image_info, 'yes')
                data.loc[idx, question]=score
            for question in best_negatives:
                #print(data)
                score= self.get_absolut_score_question_viz(question, image_info, 'no')
                data.loc[idx, question]=score 
        return data