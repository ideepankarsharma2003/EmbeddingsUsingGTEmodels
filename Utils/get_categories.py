from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import numpy as np
import json



label2id= json.load(
    open('data/categories_refined.json', 'r')
)
id2label= {}
for key in label2id.keys():
    id2label[label2id[key]] = key
 


model_name= "/home/ubuntu/SentenceStructureComparision/finetuned_entity_categorical_classification/checkpoint-3362"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)


# probabilities = 1 / (1 + np.exp(-logit_score))
def logit2prob(logit):
    # odds =np.exp(logit)
    # prob = odds / (1 + odds)
    prob= 1/(1+ np.exp(-logit))
    return np.round(prob, 3)




def predict(sentence: str):
    '''
    Returns prediction dictionary
    '''
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # print("logits: ", logits)
    predicted_class_id = logits.argmax().item()
    
    # get probabilities using softmax from logit score and convert it to numpy array
    probabilities_scores = F.softmax(logits, dim = -1).numpy()[0]
    individual_probabilities_scores = logit2prob(logits.numpy()[0])
    
    
    d= {}
    d_ind= {}
    # d_ind= {}
    for i in range(27):
        # print(f"P({id2label[i]}): {probabilities_scores[i]}")
        # d[f'P({id2label[i]})']= format(probabilities_scores[i], '.2f')
        d[f'P({id2label[i]})']= round(probabilities_scores[i], 3)
        
        
    for i in range(27):
        # print(f"P({id2label[i]}): {probabilities_scores[i]}")
        # d[f'P({id2label[i]})']= format(probabilities_scores[i], '.2f')
        d_ind[f'P({id2label[i]})']= (individual_probabilities_scores[i])
        
    d_ind['Predicted Label']=     model.config.id2label[predicted_class_id]
    d_ind['Predicted Label Score']=     individual_probabilities_scores[predicted_class_id]
    d['Predicted Label']=     model.config.id2label[predicted_class_id]
    d['Predicted Label Score']=     probabilities_scores[predicted_class_id]

    print("Predicted Class: ", model.config.id2label[predicted_class_id], f"\nprobabilities_scores: {individual_probabilities_scores[predicted_class_id]}\n")
    if d_ind['Predicted Label Score']<0.5:
        d_ind['Predicted Label']="Other"
    return d_ind
    
    
    
def get_top_labels(keyword: str):
    '''
    Returns score list
    '''
    inputs = tokenizer(keyword, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # print("logits: ", logits)
    predicted_class_id = logits.argmax().item()
    
    
    # get probabilities using softmax from logit score and convert it to numpy array
    # probabilities_scores = F.softmax(logits, dim = -1).numpy()[0]
    individual_probabilities_scores = logit2prob(logits.numpy()[0])
    
    # score_list= []
    
    # for i in range(27):
    #     score= individual_probabilities_scores[i]
    #     if score>=0.8: 
    #         score_list.append(
    #                 (id2label[i], str(score))
    #             )
    #     # if score>=0.5: 
    #     #     score_list.append(
    #     #         (id2label[i], score)
    #     #     )
            
            
    # score_list.sort(
    #     key= lambda x: x[1], reverse=True
    # )
            
    # return str(score_list[:5])
 
    prob= individual_probabilities_scores[predicted_class_id]
    label= id2label[predicted_class_id] if prob>=0.8 else "Other"
    
    return label



def get_top_labels_bulk(keywords: list):
    # category_dict= {}
    # for keyword in keywords:
    #     category_dict[keyword]= get_top_labels(keyword)
        
    # return category_dict
    for i in range(len(keywords)):
        keywords[i] = ' ' + get_top_labels(keywords[i])+' '+ keywords[i]
    return keywords


def get_top_labels_bulk_v2(keywords: list):
    # category_dict= {}
    # for keyword in keywords:
    #     category_dict[keyword]= get_top_labels(keyword)
        
    # return category_dict
    inputs = tokenizer(keywords, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits # n logits --> for all keywords
        
        
    for i in range(len(keywords)):
        keywords[i] = keywords[i]+ ' '+ id2label[logits[i].argmax().item()]
    return keywords
        
    
    