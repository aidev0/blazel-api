from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
