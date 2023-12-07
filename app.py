import uvicorn
import sys
import os
from fastapi import FastAPI, status, APIRouter, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import datetime

from main import (
    generate_base_embeddings, 
    generate_cosine_similarity,
    generate_keywords_Ngram
    )
from pydantic import BaseModel



from keys import fastapi_key



from fastapi.security import APIKeyHeader
from fastapi import Security

api_key_header= APIKeyHeader(name="Authorization")
api_keys = [
    fastapi_key
]  # This is encrypted in the database


def api_key_auth(api_key: str = Security(api_key_header)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid FAST_API_ENDPOINT_KEY"
        )




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
    
    
    

app = FastAPI()
router= APIRouter(dependencies=[Security(api_key_auth)])





class Keywords_For_Seed(BaseModel):
    seed_keyword: str
    num_keywords: int = 50
    num_urls: int = 10
    top_n: int = 7






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



@router.post('/get-similarity-against_with_ngrams')
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












@router.post('/base')
async def base(text: dict):
    
    try: 
        text= text.get("text")
        print(text)
        
        embeddings= generate_base_embeddings(text)
        print(f"embeddings: {embeddings.shape}")
        
        return JSONResponse({
            "embeddings": embeddings.tolist()
        }, media_type='application/json')
    except Exception as e:
        return Response(f'Error occured: {e}')


# for bulk processing the data

@router.post('/bulk-base')
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
    
    
    
    
    

@router.post('/get-similarity-against')
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
@router.post('/get-similarity-against-embedding')

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
        


app.include_router(router)


if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)