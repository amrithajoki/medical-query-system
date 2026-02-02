# from fastapi.responses import RedirectResponse
# from fastapi import FastAPI
# from pydantic import BaseModel
# import sys
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# import os


# # Get port from environment (Render provides this)
# PORT = int(os.getenv("PORT", 8000))
# app = FastAPI()

# @app.get("/", include_in_schema=False)
# async def docs_redirect():
#     return RedirectResponse(url='/docs')

# class Query(BaseModel):
#     text: str

# @app.post("/ask")
# async def ask_orchestrator(query: Query):
#     server_params = StdioServerParameters(
#         command=sys.executable,
#         args=["mcp_server/server.py"]
#     )

#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             # Calling the tool we decorated with @mcp.tool()
#             response = await session.call_tool("analyze_and_fetch", arguments={"query": query.text})
#             return {"answer": response.content[0].text}
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import sys
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = FastAPI()

# 1. FIXED PATHING: Calculates absolute path to the server
# Since this file is in 'api/', we go up one level to the root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_SCRIPT = os.path.join(BASE_DIR, "mcp_server", "server.py")

class Query(BaseModel):
    text: str

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

@app.post("/ask")
async def ask_orchestrator(query: Query):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env=os.environ.copy()  # Pass all environment variables
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Use query.text from the Pydantic model
                response = await session.call_tool("analyze_and_fetch", arguments={"query": query.text})
                
                # 2. FIXED DATA TYPE: Force conversion to standard string to prevent JSON errors
                answer_content = response.content[0].text
                return {"answer": str(answer_content)}
                
    except Exception as e:
        # Returns the actual error to the browser for easier debugging
        import traceback
        return {
            "error": "Internal Server Error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "debug_info": {
                "server_script": SERVER_SCRIPT,
                "executable": sys.executable,
                "cwd": os.getcwd(),
                "exists": os.path.exists(SERVER_SCRIPT)
            }
        }

@app.get("/debug/files")
async def check_files():
    """Diagnostic endpoint to check file existence on Render"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "intent_classifier.pkl")
    log_path = os.path.join(os.getcwd(), "mcp_server.log")
    
    log_content = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_content = f.read()

    return {
        "base_dir": base_dir,
        "model_exists": os.path.exists(model_path),
        "model_path": model_path,
        "log_path": log_path,
        "log_content": log_content,
        "cwd": os.getcwd(),
        "files_in_root": os.listdir(base_dir),
        "files_in_models": os.listdir(os.path.join(base_dir, "models")) if os.path.exists(os.path.join(base_dir, "models")) else []
    }
