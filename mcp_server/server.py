# from typing import Any, Dict, List
# import sys

# from mcp.server import Server
# from mcp.types import Tool, TextContent
# from mcp.server.stdio import stdio_server


# class MedicalQueryServer(Server):
#     def __init__(self):
#         super().__init__(name="Medical Query MCP Server")

#     async def list_tools(self) -> List[Tool]:
#         return [
#             Tool(
#                 name="medical_query",
#                 description="Answer medical imaging queries",
#                 inputSchema={
#                     "type": "object",
#                     "properties": {
#                         "query": {"type": "string"}
#                     },
#                     "required": ["query"]
#                 }
#             )
#         ]

#     async def call_tool(
#         self, name: str, arguments: Dict[str, Any]
#     ) -> List[TextContent]:

#         if name != "medical_query":
#             raise ValueError(f"Unknown tool: {name}")

#         query = arguments["query"]

#         return [
#             TextContent(
#                 type="text",
#                 text=f"Received medical query: {query}"
#             )
#         ]


# if __name__ == "__main__":
#     server = MedicalQueryServer()

#     stdio_server(
#         server,
#         #initialization_options={}
#     )

# import os
# from dotenv import load_dotenv
# from fastmcp import FastMCP
# import joblib
# import httpx

# # This line reads the .env file and loads the variables into your system
# load_dotenv()

# mcp = FastMCP("Medical-Query-Orchestrator")
# model = joblib.load(os.path.join(os.getcwd(), "models", "intent_classifier.pkl"))

# @mcp.tool()
# async def analyze_and_fetch(query: str) -> str:
#     # ... (ML logic remains the same) ...

#     async with httpx.AsyncClient() as client:
#         params = {
#             "db": "pubmed",
#             "term": final_search,
#             "retmode": "json",
#             # Now we use os.getenv to grab the values from your .env file
#             "email": os.getenv("NCBI_EMAIL"),
#             "api_key": os.getenv("NCBI_API_KEY")
#         }
       
        
#         # We use the search utility (esearch)
#         url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#         response = await client.get(url, params=params)
        
#         # Extract the count from the nested JSON
#         data = response.json()
#         count = data.get("esearchresult", {}).get("count", "0")

#     return f"Target Intent: {intent} | PubMed Results: {count} studies found for '{search_term}'."

# if __name__ == "__main__":
#     mcp.run()
"""
MCP Server for Medical Query Processing
Handles intent prediction, entity extraction, and PubMed API queries
"""

import pickle
import sys
import os
from pathlib import Path
import asyncio
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Project imports
from ml_model.entity_extractor import EntityExtractor

import logging
import sys
# Configure logging to stderr for Render debugging (safe for MCP)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-server")
logger.info("Server script starting...")

class MedicalQueryServer:
    """MCP Server for medical query processing"""
    
    def __init__(self):
        self.server = Server("medical-query-server")
        
        try:
            # Load ML model
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / 'models' / 'intent_classifier.pkl'
            
            logger.info(f"Looking for model at: {model_path}")
            if not model_path.exists():
                logger.error(f"Model file NOT FOUND at {model_path}")
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Please run: python ml_model/trainer.py"
                )
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            
            # Initialize entity extractor
            self.entity_extractor = EntityExtractor()
            logger.info("Entity extractor initialized")
            
            # Setup MCP tools
            self.setup_tools()
            logger.info("Tools registered")
            
        except Exception as e:
            logger.exception("Failed to initialize MedicalQueryServer")
            raise
    
    def setup_tools(self):
        """Register MCP tools"""
        
        @self.server.list_tools()
        async def list_tools():
            """Define available tools"""
            return [
                Tool(
                    name="analyze_and_fetch",
                    description="Analyze medical query and fetch results from PubMed",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language medical imaging query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            """Handle tool calls"""
            if name == "analyze_and_fetch":
                result = await self.analyze_and_fetch(arguments["query"])
                return [TextContent(type="text", text=result)]
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def analyze_and_fetch(self, query: str) -> str:
        """
        Main processing function:
        1. Predict intent
        2. Extract entities
        3. Query PubMed
        4. Return formatted result
        """
        # Step 1: Predict intent using ML model
        intent = self.model.predict([query])[0]
        
        # Step 2: Extract entities (modality, body_part)
        entities = self.entity_extractor.extract(query)
        
        # Step 3: Query PubMed API
        pubmed_result = await self.query_pubmed(intent, entities)
        
        # Step 4: Format final response
        response = {
            "query": query,
            "intent": intent,
            "entities": {
                "modality": entities['modality'],
                "body_part": entities['body_part']
            },
            "result": pubmed_result
        }
        
        # Return as formatted string
        return str(response)
    
    async def query_pubmed(self, intent: str, entities: dict) -> str:
        """Query PubMed API based on intent and entities"""
        
        # Build search query from entities
        search_terms = []
        if entities['modality']:
            search_terms.append(entities['modality'])
        if entities['body_part']:
            search_terms.append(entities['body_part'])
        
        search_query = " AND ".join(search_terms) if search_terms else "medical imaging"
        
        # Call PubMed API
        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": search_query,
                    "retmode": "json",
                    "retmax": 10
                },
                timeout=10
            )
            
            data = response.json()
            count = data.get('esearchresult', {}).get('count', 0)
            ids = data.get('esearchresult', {}).get('idlist', [])
            
        except Exception as e:
            return f"Error querying PubMed: {str(e)}"
        
        # Format response based on intent
        if intent == "study_count":
            return f"Found {count} studies matching your query."
        
        elif intent == "list_studies":
            if ids:
                return f"Study IDs: {', '.join(ids[:5])}"
            return "No studies found."
        
        elif intent == "list_modalities":
            modalities = self.entity_extractor.get_all_modalities()
            return f"Available modalities: {', '.join(modalities)}"
        
        elif intent == "list_body_parts":
            body_parts = self.entity_extractor.get_all_body_parts()
            return f"Available body parts: {', '.join(body_parts[:20])}..."
        
        return "Unable to process query."
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = MedicalQueryServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())