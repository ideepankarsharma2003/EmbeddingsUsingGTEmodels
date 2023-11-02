from http.client import HTTPSConnection
from base64 import b64encode
from json import loads
from json import dumps

class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
        connection = HTTPSConnection(self.domain)
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self.username, self.password)).encode("ascii")
                ).decode("ascii")
            headers = {'Authorization' : 'Basic %s' %  base64_bytes, 'Content-Encoding' : 'gzip'}
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        return self.request(path, 'GET')

    def post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        return self.request(path, 'POST', data_str)



client = RestClient("kumar@warewe.com", "fd5a40e08700c882")
# client = RestClient("deepankar@warewe.com", "cb1661e9ec7c1fba")


def generate_seo_metatitle(keyword, num_query_results=10):
    post_data = dict()
    # You can set only one task at a time
    post_data[len(post_data)] = dict(
        language_code="en",
        location_code=2840,
        keyword=keyword
    )
    
    response = client.post("/v3/serp/google/organic/live/regular", post_data)



    if response["status_code"] == 20000:
        # print(response)
        d= response['tasks'][0]
        # print(d)
        result_dict= d['result'][0]['items']
        summary= ''
        for i in result_dict[:num_query_results]:
            # summary+= i['title']+' '
            x= i['description']
            if x:
                # summary+= i['title']+' '+ i['domain']+' '+ i['url']+' '+x+' '
                summary+= i['title']+' '+' '+x+' '
        print(summary)
        return summary.replace('\n', ' ') 
        
        # do something with result
    else:
        print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"] , f" for keyword {keyword}"))
        

def generate_seo_metatitle_train(keyword, num_query_results=10):
    post_data = dict()
    # You can set only one task at a time
    post_data[len(post_data)] = dict(
        language_code="en",
        location_code=2840,
        keyword=keyword
    )
    
    response = client.post("/v3/serp/google/organic/live/regular", post_data)



    if response["status_code"] == 20000:
        # print(response)
        d= response['tasks'][0]
        # print(d)
        result_dict= d['result'][0]['items']
        summary= ''
        for i in result_dict[:num_query_results]:
            # summary+= i['title']+' '
            x= i['description']
            if x:
                summary+= i['title']+' '+ i['domain']+' '+ i['url']+' '+x+' '
        # print(summary)
        return summary.replace('\n', ' ') 
        
        # do something with result
    else:
        print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"] , f" for keyword {keyword}"))
        
        

def generate_top_urls(keyword, num_query_results=10)->list:
    # print(keyword)
    post_data = dict()
    # You can set only one task at a time
    post_data[len(post_data)] = dict(
        language_code="en",
        location_code=2840,
        keyword=keyword
    )
    
    response = client.post("/v3/serp/google/organic/live/regular", post_data)



    if response["status_code"] == 20000:
        # print(response)
        d= response['tasks'][0]
        # print(d)
        result_dict= d['result'][0]['items']
        urls= []
        for i in result_dict[:num_query_results]:
            # summary+= i['title']+' '
            urls.append(i['url'])
            # x= i['description']
            # if x:
            #     summary+= i['title']+' '+ i['domain']+' '+ i['url']+' '+x+' '
        # print(summary)
        # print(urls)
        return (urls)
        
        # do something with result
    else:
        print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"] , f" for keyword {keyword}"))
        
        





def generate_seo_metatitle_train_for_kg(keyword, num_query_results=10):
    post_data = dict()
    # You can set only one task at a time
    post_data[len(post_data)] = dict(
        language_code="en",
        location_code=2840,
        keyword=keyword
    )
    
    response = client.post("/v3/serp/google/organic/live/regular", post_data)



    if response["status_code"] == 20000:
        # print(response)
        d= response['tasks'][0]
        # print(d)
        result_dict= d['result'][0]['items']
        summary= ''
        for i in result_dict[:num_query_results]:
            # summary+= i['title']+' '
            x= i['description']
            if x:
                # summary+= i['title']+' '+ i['domain']+' '+ i['url']+' '+x+' '
                summary+= i['title']+' '+x+' '
        # print(summary)
        return summary.replace('\n', ' ') 
        
        # do something with result
    else:
        print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"] , f" for keyword {keyword}"))
        
        
        
def generate_intent_using_dataforseo(keyword_list):
    post_data = dict()
    # simple way to set a task
    post_data[len(post_data)] = dict(
        keywords= keyword_list,
        language_name="English"
    )
    
    response = client.post("/v3/dataforseo_labs/google/search_intent/live", post_data)
    # you can find the full list of the response codes here https://docs.dataforseo.com/v3/appendix/errors
    if response["status_code"] == 20000:
        # print(response)
        # do something with result
        d= response['tasks'][0]
        # print(d)
        result_dict= d['result'][0]['items']
        intent_dict= dict()
        for i in result_dict:
            key= i['keyword']
            value= i['keyword_intent']['label']
            intent_dict[key]= value
        return intent_dict
    else:
        print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"]))