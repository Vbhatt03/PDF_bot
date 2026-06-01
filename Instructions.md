**AFTER INSTALLING ALL REQUIRED DEPENDENCIES USING pip install -r requirements.txt in a new python virtual env**

**TO RUN THE CLI APP**

1. Run the executable file with: ./PDF_chatbot_app_final file1.pdf file2.txt file3.csv --provider ollama/gemini

**TO RUN THE UI**
1.  uvicorn server:app --reload

**TO RUN THE CLI WITHOUT THE CHATBOT APP**
1. python3 chat.py file1.pdf file2.csv file3.txt --provider ollama/gemini

**FOR USING GEMINI BACKEND**
1. create a new .env file in the directory
2. copy the contents of .env.example file
3. create a new gemini api key from https://aistudio.google.com/api-keys?project=gen-lang-client-0085148177
4. paste it in the .env file.
