import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#Importing all the libraries required 
#before running this code tensorflow,tflearn,numoy and nltk should be installed in the PC (you can do that by using pip install)
#The tensorflow version should be 1.14.0
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#Now, firslt we will open our data file and save it in the variable 'data' 
with open("intents.json") as file:
    data = json.load(file)

#now to avoid training our model eveytime we run it , we'll use try-except function of python
#we will only train the model if it has not been trained before or else we will use the already trained version
try:
    with open("data.pickle", "rb") as f:                        #opening the file having the data of our training model
        words, labels, training, output = pickle.load(f)        #getting the data from the file
#if it is not already trained then the expect part will be triggered.
except:
    words = []                  
    labels = []
    docs_x = []
    docs_y = []
#Now we will iterate in the data file and get the data so that we can use it to train our model
    for intent in data["intents"]:          #This is the key of the json file
        for pattern in intent["patterns"]:          #iterating patterns one by one
            wrds = nltk.word_tokenize(pattern)      #tokenizing the patterns using nltk library
            words.extend(wrds)                      #adding data to our list
            docs_x.append(wrds)                     #adding data to our list
            # here we will use .get() function to get the elements
            docs_y.append(intent.get("tag"))        #adding tags to our list
             
        if intent.get("tag") not in labels:
            labels.append(intent.get("tag"))        #we will add the elements for which labels are not there

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  #here we are using stemmer to take the stem part so as to smooth the process and increase the efficiency
    words = sorted(list(set(words)))        #just sorting the list , set() is to make sure that no element is reapeated


    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]   #defining a zero matrix/list

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]     # taking the stem part again 
        #now we are filling our bag list so as we can use it later in our code ( to train our model)
        for w in words:
            if w in wrds:
                bag.append(1) #if the given word is in the doc_x list then 1 will added to the list 
            else:
                bag.append(0) #if not then 0 will be added to the list

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1         

        training.append(bag)            #here we added a list-bag to our training list
        output.append(output_row)          #here we added a output-row to our output list

    #setting these lists to numpy arrays so as to use them in tflearn and tensorflow
    training = numpy.array(training)
    output = numpy.array(output)
    #opening our file to add the data
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
#resting the graph if in case there is anything left of it from the previous trainings
tensorflow.reset_default_graph()

#now we are training our model with one input layer , 3 hidden layer (each of length 8), and an ouput level or final level.
net = tflearn.input_data(shape=[None, len(training[0])])        
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #we used softmax to obtain the result in probability(i.e 0 or 1)
net = tflearn.regression(net)       

model = tflearn.DNN(net)
#Now we have succesfully trained our data.

#Now if we already have trained our data we donot need to do it again so we'll use try except function of python for that.
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
  

  #Just getting all the words in a place , i.e in bag list
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

#Here is the chat function that will at the end of processing help in the chatting of the user and the model
#or in other language the exchange of information
def chat(message):
    print("Start talking with the bot (type quit to stop)!") #It will print so that the user knows he can stop the chatbot by writing quit
    while True:     
        inp = message #inp is just the input from the user
        if inp.lower() == "quit":    #If the user wants to quit , then the file will stop and he will exit
            break

        results = model.predict([bag_of_words(inp, words)]) #We are processing the data through our trained model
        results_index = numpy.argmax(results)   #this will give the index of what should be selected.
        tag = labels[results_index]             #the label of selected index if the one we need to share/print the response of 

        for tg in data["intents"]:
            if tg.get('tag') == tag:        
                responses = tg['responses']             #this will store the response in reponses variable

        return (random.choice(responses))               #as there are multiple reponses for a single tag , we will use random to print/output a random reponse out of them