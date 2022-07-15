from transformers import ViltProcessor, ViltForQuestionAnswering, AutoTokenizer, AutoConfig, ViltForImagesAndTextClassification
import pandas as pd
import cv2 as cv
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class PreTrained_Model:
    def __init__(self, model_path_vqa, model_path_nlvr):
        self.model_path_vqa = model_path_vqa
        self.config_vqa = AutoConfig.from_pretrained(
            model_path_vqa,  output_hidden_states=True, output_attentions=True)
        self.processor_vqa = ViltProcessor.from_pretrained(model_path_vqa)
        self.model_vqa = ViltForQuestionAnswering.from_pretrained(
            model_path_vqa, config=self.config_vqa)

        self.model_path_nlvr = model_path_nlvr
        self.processor_nlvr = ViltProcessor.from_pretrained(model_path_nlvr)
        self.model_nlvr = ViltForImagesAndTextClassification.from_pretrained(
            model_path_nlvr)

    def vqa(self, question, image):
        VQ_encoding = self.processor_vqa(image, question, return_tensors="pt")
        outputs = self.model_vqa(**VQ_encoding, return_dict=True)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model_vqa.config.id2label[idx]

    def nlvr(self, question, image1, image2):
        text = question
        encoding = self.processor_nlvr(
            [image1, image2], text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_nlvr(
                input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            output = dict()
        for label, id in self.model_nlvr.config.label2id.items():
            output[label] = probs[:, id].item()
        return output['True']
    
    def shadow_remove(self, img):
        rgb_planes = cv.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv.medianBlur(dilated_img, 21)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            result_norm_planes.append(norm_img)
        shadowremov = cv.merge(result_norm_planes)
        return shadowremov

    def get_binary_accuracy_question_viz(self, question, viz_images_info, answer):
        result = 0
        for loc in viz_images_info[:5]:
            image = cv.imread(f'{loc}')
            model_answer = self.vqa(question, image)
            if model_answer == answer:
                result += 1
        sum_results = result/5
        gt = viz_images_info[5]/5
        if sum_results >= (gt-0.2) and sum_results <= (gt+0.2):
            score = 1
        else:
            score = 0
        return score

    def get_absolut_score_question_viz(self, question, viz_images_info, answer):
        result_vqa = 0
        result_nlvr = 0
        for loc in viz_images_info[:5]:
            image = cv.imread(f'{loc}')
            model_answer = self.vqa(question, image)
            if model_answer == answer:
                result_vqa += 1
        score = result_vqa/5
        return score

    def get_score_nlvr(self, question, class_name, image_info):
        score = 0
        if class_name == 'nlvr_class1':
            img2loc = 'classes_reference/class1.jpg'
        elif class_name == 'nlvr_class2':
            img2loc = 'classes_reference/class2.jpg'
        elif class_name == 'nlvr_class3':
            img2loc = 'classes_reference/class3.jpg'
        elif class_name == 'nlvr_class4':
            img2loc = 'classes_reference/class4.jpg'
        elif class_name == 'nlvr_class5':
            img2loc = 'classes_reference/class5.jpg'
        image2 = cv.imread(f'{img2loc}')
        image2_processed= self.shadow_remove(image2)
        for loc in image_info[:5]:
            image = cv.imread(f'{loc}')
            img_processed= self.shadow_remove(image)
            """ if class_name == 'nlvr_class1':
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                imgplot = plt.imshow(image)
                ax.set_title('Before')
                plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
                ax = fig.add_subplot(1, 2, 2)
                imgplot = plt.imshow(img_processed)
                imgplot.set_clim(0.0, 0.7)
                ax.set_title('After')
                plt.show() """
            model_answer = self.nlvr(question, img_processed, image2_processed)
            score += model_answer
        
        return score

    def get_relevant_questions_data(self, data, images, best_positives, best_negatives, reasoning_classes, reasoning_questions):
        for idx, row in data.iterrows():
            data.to_csv(r'D:\Projects\2022\GarbageDetection\CLEAN\main.csv', index = True, header=True)
            #if(idx > 10):
            #    break
            print(idx, 'grountruth:', row.loc_dirty, 'image1:', row.image1)
            image_info = images.get_viz_images_info(data, row.image1)
            """ for question in best_positives:
                # print(data)
                score = self.get_absolut_score_question_viz(
                    question, image_info, 'yes')
                data.loc[idx, question] = score
            for question in best_negatives:
                # print(data)
                score = self.get_absolut_score_question_viz(
                    question, image_info, 'no')
                data.loc[idx, question] = score """
            for question in reasoning_questions:
                for class_name in reasoning_classes:
                    score = self.get_score_nlvr(
                        question, class_name, image_info)
                    print(idx, question+'_'+class_name, score)
                    data.loc[idx, question+'_'+class_name] = score
        return data
