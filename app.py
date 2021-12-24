import uvicorn
from fastapi import FastAPI
from time import perf_counter 


from core.predict import Predictor 
     
app = FastAPI() 
predictor = Predictor("data")

@app.post("/predict/{sentence}")
def predict(sentence:str):

    start =  perf_counter()
    tokens, tags = predictor(sentence) 
    end =  perf_counter()
    return {"tokens" : tokens, "tags" : tags, "time_seconds" : f"{end - start:5f}"} 


if __name__ == "__main__":
    uvicorn.run(app, port = 8000)
