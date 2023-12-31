
from sentence_transformers import SentenceTransformer, util
from basic_cleaner import clean
import requests
import json
import spacy
import string
from Utils.intent_embeddings import (
    intents,
    intent_embeddings,
    reverse_intent
)
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from Utils.client import generate_seo_metatitle

import numpy as np
from summa import summarizer
import time
from keybert import KeyBERT
from keys import  *
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= gcp_credentials

from langchain.embeddings import VertexAIEmbeddings
vertex_ai_embeddings = VertexAIEmbeddings()


model_base = SentenceTransformer('thenlper/gte-base', device='cuda')
# model_base = SentenceTransformer('TaylorAI/gte-tiny', device='cuda')
# model_base = SentenceTransformer('thenlper/gte-small', device='cuda')
kw_model = KeyBERT(model_base)

# model_base = SentenceTransformer('hkunlp/instructor-large', device='cuda')
model_bge_large= model_base
model_large= model_base
# model_large = SentenceTransformer('thenlper/gte-large', device='cuda')
# model_bge_large = SentenceTransformer('BAAI/bge-large-en', device='cuda')
# model_e5_large_v2 = SentenceTransformer('efederici/e5-large-v2-4096', {"trust_remote_code": True})

# model_e5_large_v2.max_seq_length= 4096

# tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
# model = AutoModel.from_pretrained('thenlper/gte-base').to("cuda")

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def str_2_list_of_str(s):
    """
    Convert a string to a list of strings.
    """
    s= s.replace('[', '')
    s= s.replace(']', '')
    s= s.replace('\n', '')
    s= s.replace('\t', '')
    s= s.replace('  ', '')
    s= s.replace('"', '')
    s= s.replace("'", '')
    list_of_strings= s.split(',')
    return list_of_strings



def generate_palm_embeddings(text: list):
    return vertex_ai_embeddings.embed_documents(text)





def generate_base_embeddings(text): 
    """
    Generate embeddings for the given text using GTE-base.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    embeddings= model_base.encode(text, batch_size=32, convert_to_tensor=True)
    # embeddings= model_base.encode(text, batch_size=12000,convert_to_numpy=True)
    
    
    # return util.cos_sim(embeddings[0], embeddings[1])
    print("created embeddings of shape: ", embeddings.shape, flush=True)
    return embeddings.cpu().numpy()
    return embeddings


# def generate_base_embeddings_v2(text): 
#     """
#     Generate embeddings for the given text using GTE-base.
#     """
#     # for i in range(len(text)):
#     #     text[i]= clean(text[i])
#     #     print(text[i])
#     # print()
#     # embeddings= model_base.encode(text, convert_to_tensor=True)
#     batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

#     # outputs = model(**batch_dict)
#     outputs = model(**batch_dict.to("cuda"))
    
#     embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().detach().numpy()
    
    
#     # return util.cos_sim(embeddings[0], embeddings[1])
#     print("created embeddings of shape: ", embeddings.shape, flush=True)
#     # return embeddings.cpu().numpy()
#     return embeddings





'''
def generate_e5_large_v2_embeddings(text): 
    """
    Generate embeddings for the given text using e5_large_v2.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    embeddings= model_e5_large_v2.encode(text, convert_to_tensor=True)
    
    
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()

'''






def generate_large_embeddings(text):
    """
    Generate embeddings for the given text using GTE-large.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    embeddings= model_large.encode(text, convert_to_tensor=True)
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()


def generate_bge_large_embeddings(text):
    """
    Generate embeddings for the given text using BGE-large.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    embeddings= model_bge_large.encode(text, convert_to_tensor=True)
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()






def generate_cosine_similarity(e1, e2,precision:int=None):
    """
    Generate cosine similarity for the given embeddings.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    # embeddings= model_bge_large.encode(text, convert_to_tensor=True)
    # # return util.cos_sim(embeddings[0], embeddings[1])
    # return embeddings.cpu().numpy()
    if(precision > 0):
        return np.round(util.cos_sim(e1,e2).cpu().numpy().tolist(),precision).tolist()
    
    return util.cos_sim(e1, e2)

   
   
   


def generate_keyword_summary(keyword):
    """Generate a summary of the keyword"""
    response= requests.api.get(f'https://2qq35q1je7.execute-api.us-east-1.amazonaws.com/?search={keyword}')
    d= json.loads(response.text)
    
    data= d['data']
    results= data['results']
    
    s= ""
    

    for i in results:
        s+=i['url']+' '
        s+=i['description']
        
    s= s.replace("https://", '')
    s= s.replace("/", '')
    s= s.replace(",", '')
    s= s.replace("www.", '')
        
    summary=summarizer.summarize(s, words=200).replace('\n', ' ')
    # summary= spacy_tokenizer(s)
    
    return summary 
   
   


def generate_keyword_summary_for_intent(keyword):
    """Generate a summary of the keyword"""
    response= requests.api.get(f'https://2qq35q1je7.execute-api.us-east-1.amazonaws.com/?search={keyword}')
    d= json.loads(response.text)
    
    data= d['data']
    results= data['results']
    
    s= ""
    

    for i in results[:5]:
        s+=i['url']+' '
        s+=i['description']
        
    s= s.replace("https://", '')
    s= s.replace("/", '')
    s= s.replace(",", '')
    s= s.replace("www.", '')
        
    summary=summarizer.summarize(s, words=200).replace('\n', ' ')
    # summary= spacy_tokenizer(s)
    
    return summary 


def generate_keyword_summary_for_intent_v2(keyword):
    """Generate a summary of the keyword"""
    response= requests.api.get(f'https://7t4h0oe8be.execute-api.us-east-1.amazonaws.com/?search={keyword}')
    d= json.loads(response.text)
    
    data= d['items']
    results= data['results']
    
    s= ""
    

    for i in results[:5]:
        s+=i['url']+' '
        s+=i['text']
        
    s= s.replace("https://", '')
    s= s.replace("/", '')
    s= s.replace(",", '')
    s= s.replace("www.", '')
        
    summary=summarizer.summarize(s, words=200).replace('\n', ' ')
    # summary= spacy_tokenizer(s)
    
    return summary 
    




punctuations = string.punctuation
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)
    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    sentence = " ".join(mytokens)
    # return preprocessed list of tokens
    return sentence





def generate_intent(keyword):
    s_i= generate_keyword_summary_for_intent(keyword)
    e_i= generate_base_embeddings(s_i)

    cos_similarity= generate_cosine_similarity(intent_embeddings, e_i)
    dominant_intent= reverse_intent[int(np.argmax(cos_similarity))]
    score= cos_similarity[int(np.argmax(cos_similarity))]
    # print(f'dominant_intent= {dominant_intent}')
    # print(cos_similarity, '\n\n')
    print(keyword+':\n', cos_similarity, end='\n\n')
    return dominant_intent, float(score), cos_similarity



def generate_intent_v2(keyword):
    s_i= generate_seo_metatitle(keyword)
    e_i= generate_base_embeddings(s_i)

    cos_similarity= generate_cosine_similarity(intent_embeddings, e_i)
    dominant_intent= reverse_intent[int(np.argmax(cos_similarity))]
    score= cos_similarity[int(np.argmax(cos_similarity))]
    # print(f'dominant_intent= {dominant_intent}')
    # print(cos_similarity, '\n\n')
    print(keyword+':\n', cos_similarity, end='\n\n')
    return dominant_intent, float(score), cos_similarity








def generate_keywords_Ngram(
    keywords_in: list[str],
    num_keywords: int=50,
    top_n: int=4,
    start_time=time.time()
):
    
    
    metatitle= ' '+ ' \n'.join(keywords_in)+ ' '
    keywords_list= []
    for i in range(1, top_n):
        keywords = kw_model.extract_keywords(metatitle, 

                                     keyphrase_ngram_range=(1, i), 

                                    #  stop_words='english', 

                                     highlight=False,
                                     
                                    #  use_mmr=True, 
                                     
                                    #  diversity=0.7,

                                    #  top_n=30,
                                     
                                    #  nr_candidates=40
                                     top_n=num_keywords,                                     
                                     nr_candidates=num_keywords*3
                                     )

        keywords_list+= keywords
    
        
    print("--- %2.6s seconds ---[GENERATED KEYWORD LIST]" % (time.time() - start_time), flush=True)
    
    keywords_list= list(set(keywords_list))
    print("--- %2.6s seconds ---[GENERATED KEYWORD LIST-DEDUPED]" % (time.time() - start_time), flush=True)
    # keywords_list= [list(i) for i in keywords_list]
    max_count= 1
    for i in range(len(keywords_list)):
        temp= list(keywords_list[i])
        
        mc_i= 0
        keyword_i= temp[0]
        if len(keyword_i.split(' '))>1:
            mc_i= metatitle.count(keyword_i)
        else:
            
            mc_i= metatitle.count(
                    ' '+ keyword_i+' ')
            mc_i+= metatitle.count(keyword_i+'\n')
        
        # mc_i+= keywords_in.count(temp[0])
        
        temp.append(
            mc_i
        )
        max_count= max(max_count, mc_i)
        keywords_list[i]= temp
    
    print("--- %2.6s seconds ---[GENERATED KEYWORD LIST-COUNT]" % (time.time() - start_time), flush=True)
    return sorted(keywords_list,
                    key=lambda x: x[1] if (x[2]>2 and x[1]>0.8) else (x[2]/max_count)-0.25,
                    reverse=True)[:num_keywords]
        