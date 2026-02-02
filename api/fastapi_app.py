# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import os

# # ----------------------------
# # FastAPI instance (must be named 'app')
# # ----------------------------
# app = FastAPI(
#     title="Medical Query Intent API",
#     description="Predicts intent from medical text queries",
#     version="1.0.0"
# )

# # ----------------------------
# # Load trained model
# # ----------------------------
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/intent_classifier.pkl")
# model = joblib.load(MODEL_PATH)

# # ----------------------------
# # Request / Response schemas
# # ----------------------------
# class QueryRequest(BaseModel):
#     query: str

# class PredictionResponse(BaseModel):
#     query: str
#     intent: str
#     confidence: float

# # ----------------------------
# # Prediction endpoint
# # ----------------------------
# @app.post("/predict", response_model=PredictionResponse)
# def predict_intent(request: QueryRequest):
#     query_text = request.query

#     # Predict probabilities
#     probs = model.predict_proba([query_text])[0]
#     classes = model.classes_

#     # Get best prediction
#     best_idx = int(np.argmax(probs))
#     intent = classes[best_idx]
#     confidence = float(probs[best_idx])

#     return {
#         "query": query_text,
#         "intent": intent,
#         "confidence": round(confidence, 4)
#     }
# @app.get("/")
# def root():
#     return {"message": "Medical Query Intent API is running. Use /docs to see Swagger UI."}

from fastapi import FastAPI
from pydantic import BaseModel
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from fastapi.responses import RedirectResponse

# Get port from environment (Render provides this)
PORT = int(os.getenv("PORT", 8000))
app = FastAPI()
@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask_orchestrator(query: Query):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server/server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Calling the tool we decorated with @mcp.tool()
            response = await session.call_tool("analyze_and_fetch", arguments={"query": query.text})
            return {"answer": response.content[0].text}