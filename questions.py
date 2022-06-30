class Questions:
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives 
       
    def get_best_questions(self, data, images, model, k):
        scores_positives = []
        scores_negatives = []
        size_data= len(data)
        for question in self.positives:
            print(question)
            viz_scores=0
            for idx, row in data.iterrows():
                print(idx)
                image_info = images.get_viz_images_info(data, row.image1)
                viz_scores+= self.get_binary_accuracy_question_viz(question, image_info, model, 'yes')
            question_score= viz_scores/size_data
            print(question_score)
            scores_positives.append(question_score)
        for question in self.negatives:
            print(question)
            viz_scores=0
            for idx, row in data.iterrows():
                print(idx)
                image_info = images.get_viz_images_info(data, row.image1)
                viz_scores+= self.get_binary_accuracy_question_viz(question, image_info, model, 'no')
            question_score= viz_scores/size_data
            print(question_score)
            scores_negatives.append(question_score)
            
        #TODO order by score DESC    
        """ best_positives = scores_positives[:k]   
        best_negatives = scores_negatives[:k] """
        scores_positives = {
            'question': self.positives,
            'scores_positives': scores_positives,   
        }
        scores_negatives = {
            'question': self.negatives,
            'scores_negatives': scores_negatives,   
        }
        return scores_positives, scores_negatives
    
    
    def get_combos(self, positives, negatives):
        combos=[]
        for positive in positives:
            for negative in negatives:
                comb=[]
                comb.append(positive)
                comb.append(negative)
                combos.append(comb)
        return combos
            