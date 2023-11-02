
from Utils.clientv2 import generate_seo_metatitle
# from utils.cleaner import extract_paragraphs
from keybert import KeyBERT
# kw_model = KeyBERT(model='thenlper/gte-base')
kw_model = KeyBERT(model='intfloat/multilingual-e5-large')



def generate_keywords_around_seed(
    seed_keyword:str,
    num_keywords: int= 50,
    num_urls: int=10,
    top_n: int= 7
):
    metatitle= generate_seo_metatitle(seed_keyword, num_query_results=num_urls)
    keywords_list= []
    for i in range(1, top_n):
        keywords = kw_model.extract_keywords(metatitle, 

                                     keyphrase_ngram_range=(1, i), 

                                     stop_words='english', 

                                     highlight=False,
                                     
                                    #  use_mmr=True, 
                                     
                                    #  diversity=0.7,

                                     top_n=2*num_keywords,
                                     
                                     nr_candidates=3*num_keywords
                                     )

        keywords_list+= keywords
    
    keywords_list= list(set(keywords_list))
    return ( sorted(keywords_list,
                    key=lambda x: x[1],reverse=True)[:num_keywords], 
            
            
            
            )