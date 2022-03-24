# Perspective system to determine if a comment is offensive or not based on its toxicity levels
import sys
import json
import requests
import csv
import time 

# a Perspective class object which allows query tests
class Perspective:

    def __init__(self, threshold = 0.5):
        self.threshold = threshold
        api_key = ''
        self.url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
                      '?key=' + api_key)


    def predict(self, test_query):
        text = test_query.lower().strip()
        request = createRequestDict(text)
        
        # sleep 2 second to comply with perspective api
        time.sleep(2)

        success = False

        response = ""
        response_dict = {}

        while(not success):
            # send request to google perspective api
            try:
                response = requests.post(url=self.url, data=json.dumps(request))
                response_dict = json.loads(response.content)
                success = True
            except:
                # sleep 5 minutes if no response from server
                time.sleep(300)
            
        if('attributeScores' in response_dict):
            toxicity_level = response_dict['attributeScores']['TOXICITY']['summaryScore']['value']
        else:
            toxicity_level = 0.0

        if(toxicity_level > self.threshold):
            prediction = 'OFF'
        else:
            prediction = 'NOT'

        return prediction, toxicity_level
        

    # allow multiple queries in form of list
    def predictMultiple(self, test_queries):
        predictions = []
        probs = []
        for test in test_queries:
            pred, prob = self.predict(test)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs


# create a dicitonary to be sent to the Google perspective api
def createRequestDict(text):
    data_dict = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}
    }
    
    return data_dict
    
# Predicts a comment as Offensive if toxicity is greater than threshold
def PerspectivePredict(testFile, threshold = 0.5, outfile = None):
    api_key = ''
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
                      '?key=' + api_key)

    threshold = float(threshold)

    if(outfile):
        output = open(outfile, 'w')
    else:
        output = open('perspectivePredictionsOut', 'w')
    
    # walk through test file and predict offensive or not
    with open(testFile, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            text = tweet[1].lower().strip()
            request = createRequestDict(text)
            
            # sleep 2 seconds to comply with perspective api 
            time.sleep(2)
            # send request to google perspective api
            response = requests.post(url=url, data=json.dumps(request))
            response_dict = json.loads(response.content)
            
            if('attributeScores' in response_dict):
                toxicity_level = response_dict['attributeScores']['TOXICITY']['summaryScore']['value']
            else:
                toxicity_level = 0.0

            if(toxicity_level > threshold):
                prediction = 'OFF'
            else:
                prediction = 'NOT'

            output.write(tweet[0] + ',' + prediction + '\n')
        
        output.close()

if(__name__ == "__main__"):
    PerspectivePredict(sys.argv[1], sys.argv[2], sys.argv[3])
