# QuickLocalTranslator

Download ollama https://ollama.com/
Open up cmd and type ollama run llama3

Once downloaded modify the Modelfile attached how you like

Then go to the directory where it is stored and the folder/ directory above that type
ollama create Bob -f ./Modelfile

Now you have a Bob translator for the language llama3 supports

the code attached allows you to use gpt4o for vision by pressing the number 1 on the keypad
 it then takes the description and feeds it to Bob the model you have built
 and Bob dictates it to you

The number 2 uses Claude sonnet3.5

and the * key allows you to have a conversation as normal with the chatbot.

All pip installs should be in the requirement.txt
pip install -r requirements.txt
