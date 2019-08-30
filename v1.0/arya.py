import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import random
import tflearn
import json
import pickle
import speech_recognition as sr

def read_intents(path):
    ## Fetch the training data
    with open(path) as file:
        data = json.load(file)
    return data

def pre_processing(data):
    
    try: 
    	with open("data.pickle", "rb") as f:
    		words, labels, training, output = pickle.load(f)
    
    except:
    	##Tokenize
    	words = []
    	docs_x = []
    	docs_y = []
    	labels = []
    
    	for intent in data["intents"]:
    		for pattern in intent["patterns"]:
    			wrds = nltk.word_tokenize(pattern)
    			print("wrds : =====> ")
    			print(wrds)
    			words.extend(wrds)
    			print("words : =====>")
    			print(words)
    			docs_x.append(wrds)
    			print("docs_x : =====>")
    			print(docs_x)
    			docs_y.append(intent["tag"])
    			print("docs_y : =====>")
    			print(docs_y)
    
    		if intent["tag"] not in labels:
    			labels.append(intent["tag"])
    			print("labels : =====>")
    			print(labels)
    
    	words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    	words = sorted(list(set(words)))
    
    	labels = sorted(labels)
    
    	training = []
    	output = []
    
    	out_empty = [0 for _ in range(len(labels))]
    
    	for x, doc in enumerate(docs_x):
    		bag = []
    
    		wrds = [stemmer.stem(w) for w in doc]
    
    		for w in words:
    			if w in wrds:
    				bag.append(1)
    			else:
    				bag.append(0)
    
    		output_row = out_empty[:]
    		output_row[labels.index(docs_y[x])] =1
    
    		training.append(bag)
    		output.append(output_row)
    
    	training = numpy.array(training)
    	output = numpy.array(output)
    
    	with open("data.pickle", "wb") as f:
    		pickle.dump((words, labels, training, output), f)
    
    tensorflow.reset_default_graph()
    
    
    net = tflearn.input_data(shape = [None, len(training[0])])
    net = tflearn.fully_connected(net, 8 )
    net = tflearn.fully_connected(net, 8 )
    net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
    net = tflearn.regression(net)
    
    model = tflearn.DNN(net)
    
    #try:
    #	model.load("model.tflearn")
    #except:
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")
    
    return model, words, labels


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.
    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source) # #  analyze the audio source for 1 second
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #   update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable/unresponsive"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response



def chat(model, words, labels):
    import speech_recognition as sr
    print("Start talking with Arya - (Type quit to stop)!")
    while True:
        recognizer = sr.Recognizer()
        mic = sr.Microphone(device_index=1)
        response = recognize_speech_from_mic(recognizer, mic)
        
        if (response['transcription'] == None):
            print("  Arya: I didn't get that, try some other...")
        
        else:
            print("   You: " + response['transcription'])
            inp = response['transcription']
            if inp.lower() == "quit":
                break
            
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            
            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                
                print("  Arya: " + random.choice(responses))
            
            else:
                print("  Arya: I didn't get that, try some other...")
            
            
#Driver Code:
if __name__ == "__main__":
    data = read_intents("intents.json")
    
    model, words, labels = pre_processing(data)
    
    chat(model, words, labels)
            
            
