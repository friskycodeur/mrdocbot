from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
#Here we will import the chat function from main.py
from main import chat

#connection to server
class ChatServer(WebSocket):

    def handleMessage(self):
        # echo message back to client
        message = self.data
        #Get the written message by user and give it as input to chat()
        response = chat(message)
        #Print the response of from the chat()
        self.sendMessage(response)

    def handleConnected(self):
        #print connected when connected to server
        print(self.address, 'connected')

    def handleClose(self):
        #print closed when the server is closed
        print(self.address, 'closed')



server = SimpleWebSocketServer('', 8000, ChatServer)
server.serveforever()
