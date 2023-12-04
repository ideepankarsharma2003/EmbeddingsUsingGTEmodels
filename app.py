import uvicorn
import sys
import os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
# from Utils.get_categories import get_top_labels, get_top_labels_bulk_v2
# from Utils.get_keywords_utils import generate_keywords_around_seed
# from Utils.get_intent_bert_basedANN import get_intent_bulk_v2
import time
import datetime

from main import (
    generate_base_embeddings, 
    generate_palm_embeddings,
    # generate_base_embeddings_v2, 
    generate_large_embeddings, 
    str_2_list_of_str, 
    generate_bge_large_embeddings,
    # generate_e5_large_v2_embeddings,
    generate_intent    ,
    generate_cosine_similarity,
    generate_keywords_Ngram
    )
import json
from pydantic import BaseModel

class Keyword(BaseModel):
    keyword: str 

class Keyword_bulk(BaseModel):
    keyword: list

class SimilarityAgainst(BaseModel):
    main_entity: str
    compare_with_entitites: list[str]
    need_intent: bool = False

class SimilarityAgainst_with_NGrams(BaseModel):
    main_entity: str
    compare_with_entitites: list[str]
    need_intent: bool = False
    need_ngrams: bool = True
    # keywords: list[str]= [""]
    num_keywords: int= 50
    top_n: int= 4
    
    
    
class Keywords_For_Seed(BaseModel):
    seed_keyword: str
    num_keywords: int = 50
    num_urls: int = 10
    top_n: int = 7






app= FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("initializing app\n")

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')
    # return "Hello world!"

 
# @app.post('/get_keywords_for_seedkeyword')
# async def get_keywords_for_seedkeyword(obj:Keywords_For_Seed):
    
#     try: 
#         # text= str_2_list_of_str(text)
#         return generate_keywords_around_seed(
#             seed_keyword=obj.seed_keyword,
#             num_keywords=obj.num_keywords,
#             num_urls=obj.num_urls,
#             top_n=obj.top_n
#         )
#     except Exception as e:
#         return Response(f'Error occured: {e}')
    
    


@app.post('/get-similarity-against_with_ngrams')
async def get_similarity_against_with_ngrams(simag: SimilarityAgainst_with_NGrams):
    try:
        print(f"start time [ GET SIMILARITY AGAINST WITH NGRAMS]: {datetime.datetime.now()}")
        start_time = time.time()
        main_entity = simag.main_entity
        ngrams= []
        # main_entity=get_top_labels(main_entity)+ ' '+ main_entity
        print(f'main_entity: {main_entity}', flush=True)
        
        compare_with_entitites = simag.compare_with_entitites # list of strings
        # compare_with_entitites= get_top_labels_bulk_v2(compare_with_entitites)
        print(f'len compare_with_entitites: {len(compare_with_entitites)}', flush=True)
        
        # simag.need_ngrams= False
        
        print(f"need n-grams: '{simag.need_ngrams}'", flush=True)
        if simag.need_ngrams:
            ngrams= generate_keywords_Ngram(
                simag.compare_with_entitites,
                num_keywords=simag.num_keywords,
                top_n= simag.top_n,
                start_time=start_time
            )
            print("--- %2.6s seconds [GENERATED NGRAMS]---" % (time.time() - start_time))
            
        
        
        
        allkeywords= [main_entity]+compare_with_entitites
        print("--- %2.6s seconds [TRYING TO GENERATE EMBEDDINGS]---" % (time.time() - start_time))
        embeddings= generate_base_embeddings(allkeywords)
        # embeddings= []
        # for i in range(0, len(allkeywords), 420):
            
        #     embeddings+= generate_palm_embeddings(allkeywords[i:i+420])
        
        print("--- %2.6s seconds [GENERATED EMBEDDINGS]---" % (time.time() - start_time))
        
        intent=[]
        if  simag.need_intent:
            for i in range(0, len(allkeywords), 10000):
            
                keywords= allkeywords[i:i+10000]
                # print(keywords)
                
                # intent+=get_intent_bulk_v2(keywords)
                print(f"i= {i}, generated intents of shape: ", len(intent))
                # embeddings+=generate_base_embeddings(keywords)
                
                
            print("len intent: ", len(intent), flush=True)
        
        # similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=2)
        similarity_score = generate_cosine_similarity(embeddings[0],embeddings[1:],precision=2)
        # similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=-2)
        print("len similarity_score: ", len(similarity_score[0]), flush=True)
        print("--- %2.6s seconds [GENERATED SIMILARITY SCORE] ---\n\n" % (time.time() - start_time))
        return {
            "similarity": similarity_score[0],
            "intent": intent,
            "ngrams": ngrams
        }
    
    except Exception as e:
        print(e)
        return Response(f'Error occured: {e}')












@app.post('/base')
async def base(text: dict):
    
    try: 
        text= text.get("text")
        # print(type(text))
        
        # text= str_2_list_of_str(text)
        # text= text.split(',')
        # print("Converted the string to list of urls: ",text)
        
        
        # print(type(text))
        # print(text)
        
        
        
        # print(f"n_urls: {len(text)}")
        
        # text= text+' '+ get_top_labels(text)
        print(text)
        
        embeddings= generate_base_embeddings(text)
        # embeddings= embeddings.reshape(1, -1)
        # # print(embeddings.shape)
        print(f"embeddings: {embeddings.shape}")
        
        # # print(embeddings)
        # # return embeddings.tolist()
        # # return (embeddings[0][0].item())
        # # return {"text": text}
        return JSONResponse({
            "embeddings": embeddings.tolist()
        }, media_type='application/json')
    except Exception as e:
        return Response(f'Error occured: {e}')


# for bulk processing the data

@app.post('/bulk-base')
async def bulk_base(text:dict):
    try:
        text_list = text.get('text')
        
        # text_list= get_top_labels_bulk_v2(text_list)

        print(text)
        print(text_list)
        embeddings = generate_base_embeddings(text_list)
        print("Bulk embeddings: ",embeddings.shape)
        return {
        "count": embeddings.shape[0],
        "embeddings": embeddings.tolist()
        }
    except Exception as e:
        return Response("some error occured",str(e))
    

# for getting the cosine similarity

# @app.post('/get-similarity-against')
# async def get_similarity_against(simag: SimilarityAgainst):
#     try:
#         main_entity = simag.main_entity
#         # main_entity=get_top_labels(main_entity)+ ' '+ main_entity
#         print(f'main_entity: {main_entity}', flush=True)
        
#         compare_with_entitites = simag.compare_with_entitites # list of strings
#         # compare_with_entitites= get_top_labels_bulk_v2(compare_with_entitites)
#         print(f'len compare_with_entitites: {len(compare_with_entitites)}', flush=True)
        
#         intent= []

#         embeddings= generate_base_embeddings(
#             [main_entity]+compare_with_entitites
#         )
#         for keywords in [main_entity]+compare_with_entitites
#         intent= get_intent_bulk_v2(compare_with_entitites)
#         print("len intent: ", len(intent), flush=True)
        
#         # similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=2)
#         similarity_score = generate_cosine_similarity(embeddings[0],embeddings[1:],precision=2)
#         # similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=-2)
#         print("len similarity_score: ", len(similarity_score[0]), flush=True)
#         return {
#             "similarity": similarity_score[0],
#             "intent": intent
#         }
    
#     except Exception as e:
#         return Response(f'Error occured: {e}')








@app.post('/get-similarity-against')
async def get_similarity_against(simag: SimilarityAgainst):
    try:
        # print(f"start time: {time.time()}")
        start_time = time.time()
        main_entity = simag.main_entity
        # main_entity=get_top_labels(main_entity)+ ' '+ main_entity
        print(f'main_entity: {main_entity}', flush=True)
        
        compare_with_entitites = simag.compare_with_entitites # list of strings
        # compare_with_entitites= get_top_labels_bulk_v2(compare_with_entitites)
        print(f'len compare_with_entitites: {len(compare_with_entitites)}', flush=True)
        
        allkeywords= [main_entity]+compare_with_entitites
        
        
        
        embeddings= generate_base_embeddings(allkeywords)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        
        intent=[]
        if  simag.need_intent:
            for i in range(0, len(allkeywords), 10000):
            
                keywords= allkeywords[i:i+10000]
                # print(keywords)
                
                # intent+=get_intent_bulk_v2(keywords)
                print(f"i= {i}, generated intents of shape: ", len(intent))
                # embeddings+=generate_base_embeddings(keywords)
                
                
            print("len intent: ", len(intent), flush=True)
        
        # similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=2)
        similarity_score = generate_cosine_similarity(embeddings[0],embeddings[1:],precision=2)
        # similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=-2)
        print("len similarity_score: ", len(similarity_score[0]), flush=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        return {
            "similarity": similarity_score[0],
            "intent": intent
        }
    
    except Exception as e:
        return Response(f'Error occured: {e}')


















        
        """ 
        {
"main_entity":[
    -0.025111692026257515,
    -0.006442782003432512
  ],
"compare_with": ["hey"]
}
        """
        
import numpy as np
@app.post('/get-similarity-against-embedding')

async def get_similarity_against(text:dict):
    try:
        main_entity_embedding = text.get("main_entity")
        main_entity_embedding = np.array(main_entity_embedding,dtype=np.float32)
    
        compare_with_entitites = text.get("compare_with") # list of strings
        to_compare_entitites_embedding = generate_base_embeddings(compare_with_entitites)
        
        similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=2)

        return {
            "similarity": similarity_score[0]
        }
    
    except Exception as e:
        return Response(f'Error occured: {e}',status_code=400)
        
    

    
# @app.post('/large')
# async def large(text:dict):
    
#     try: 
#         # text= str_2_list_of_str(text)
#         text= text.get("text")
        
#         embeddings= generate_large_embeddings(text)
#         # embeddings= embeddings.reshape(1, -1)
        
#         print(f"n_urls: {len(text)}")
#         print(f"embeddings: {embeddings.shape}")

#         # return (embeddings[0][0].item())
#         return JSONResponse({
#             "embeddings": embeddings.tolist()
#         }, media_type='application/json')
#     except Exception as e:
#         return Response(f'Error occured: {e}')


'''
@app.post('/large')
async def large(text:dict):
    
    try: 
        # text= str_2_list_of_str(text)
        text= text.get("text")
        
        embeddings= generate_large_embeddings(text)
        # embeddings= embeddings.reshape(1, -1)
        
        print(f"n_urls: {len(text)}")
        print(f"embeddings: {embeddings.shape}")

        # return (embeddings[0][0].item())
        return JSONResponse({
            "embeddings": embeddings.tolist()
        }, media_type='application/json')
    except Exception as e:
        return Response(f'Error occured: {e}')



@app.post('/bgelarge')
async def large(text:dict):
    
    try: 
        # text= str_2_list_of_str(text)
        text= text.get("text")
        
        embeddings= generate_bge_large_embeddings(text)
        # embeddings= embeddings.reshape(1, -1)
        
        print(f"n_urls: {len(text)}")
        print(f"embeddings: {embeddings.shape}")

        # return (embeddings[0][0].item())
        return JSONResponse({
            "embeddings": embeddings.tolist()
        }, media_type='application/json')
    except Exception as e:
        return Response(f'Error occured: {e}')
        # return Response(f'Error occured: {e}')

'''

'''
@app.get('/intent')
async def large(text):
    
    try: 
        # text= str_2_list_of_str(text)
        # text= text.get("text")
        
        intent, score, similarity= generate_intent(text)
        # embeddings= embeddings.reshape(1, -1)
        
        # print(f"n_urls: {len(text)}")
        # print(f"embeddings: {embeddings.shape}")

        # return (embeddings[0][0].item())
        return intent, score
    except Exception as e:
        return Response(f'Error occured: {e}')
        # return Response(f'Error occured: {e}')

'''

# @app.post('/e5_large_v2')
# async def model_e5_large_v2(text:dict):
    
#     try: 
#         # text= str_2_list_of_str(text)
#         text= text.get("text")
        
#         embeddings= generate_e5_large_v2_embeddings(text)
#         # embeddings= embeddings.reshape(1, -1)
        
#         print(f"n_urls: {len(text)}")
#         print(f"embeddings: {embeddings.shape}")

#         # return (embeddings[0][0].item())
#         return JSONResponse({
#             "embeddings": embeddings.tolist()
#         }, media_type='application/json')
#     except Exception as e:
#         return Response(f'Error occured: {e}')

'''
    
@app.post('/get_category')
async def get_category(text:Keyword):
    
    try: 
        # text= str_2_list_of_str(text)
        text= text.keyword
        
        # category= get_top_labels(text)

        # return (embeddings[0][0].item())
        # return category
        return JSONResponse({
            "categories": (category)
        }, media_type='application/json')
    except Exception as e:
        return Response(f'Error occured: {e}')
    
    
@app.post('/get_category_bulk')
async def get_category_bulk(text:Keyword_bulk):
    
    try: 
        # text= str_2_list_of_str(text)
        text= text.keyword
        print(text)
        
        category= get_top_labels_bulk_v2(text)
        print(category)

        # return (embeddings[0][0].item())
        return category
        # return JSONResponse(
        #                     category, media_type='application/json')
    except Exception as e:
        return Response(f'Error occured: {e}')
'''


if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)