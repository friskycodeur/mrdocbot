# mrdocbot
A interactive chat-bot for information about Covid-19/Corona Virus.
#here are the steps to properly run this chat-bot

To run the program follow the given steps:-
1.Ensure that you are using python version 3.6.
2.Then install the required library using pip command:-
  1)nltk
  2)numpy
  3)tflearn
  4)tensorflow==1.14.0
  5)SimpleWebSocketServer
3.Download all the programs main.py,server.py,intents.json,index.html in the same folder.
4.Afterthat run the program main.py.
(While the program is being processed if the error "ImportError: DLL load failed: The specified procedure could not be found." occurs
then use this command pip install pip install protobuf==3.6.0 to degrade the version of protobuf==3.6.1 to 3.6.0) 
5.After the model is trained run server.py.
6.Then open index.html in the web browser and chat with the bot!
