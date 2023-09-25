import uvicorn
import sys
import os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from main import (
    generate_base_embeddings, 
    generate_large_embeddings, 
    str_2_list_of_str, 
    generate_bge_large_embeddings,
    # generate_e5_large_v2_embeddings,
    generate_intent    ,
    generate_cosine_similarity
    )
import json





app= FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("initializing app")

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')
    # return "Hello world!"


 



@app.post('/base')
async def base(text: dict):
    
    try: 
        text= text.get("text")
        # print(type(text))
        print(text)
        
        # text= str_2_list_of_str(text)
        # text= text.split(',')
        # print("Converted the string to list of urls: ",text)
        
        
        # print(type(text))
        # print(text)
        
        
        
        # print(f"n_urls: {len(text)}")
        
        
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

@app.post('/get-similarity-against')
async def get_similarity_against(text:dict):
    try:
        main_entity = text.get("main_entity")
        compare_with_entitites = text.get("compare_with") # list of strings
        
        main_entity_embedding = generate_base_embeddings(main_entity)
        to_compare_entitites_embedding = generate_base_embeddings(compare_with_entitites)
        
        similarity_score = generate_cosine_similarity(main_entity_embedding,to_compare_entitites_embedding,precision=2)
        return {
            "similarity": similarity_score[0]
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



if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
