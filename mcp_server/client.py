import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    # 1. Define how to launch your server
    # We tell the client to run the same python command you used
    server_params = StdioServerParameters(
        command=sys.executable,  # Uses the current python.exe
        args=["mcp_server/server.py"], # Path to your server script
        env=None
    )

    print("🔌 Connecting to server...")
    
    # 2. Connect via Standard Input/Output
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # 3. Initialize the connection (The Handshake)
            await session.initialize()
            print("✅ Handshake successful!")

            # 4. List Available Tools
            tools = await session.list_tools()
            print(f"\n🛠️  Found {len(tools.tools)} tool(s):")
            for tool in tools.tools:
                print(f"   - {tool.name}: {tool.description}")

            # 5. Call the Tool
            print("\n🧪 Testing 'medical_query' tool...")
            result = await session.call_tool(
                name="medical_query",
                arguments={"query": "patient has frequent headaches"}
            )

            # 6. Show the Result
            print(f"📄 Result: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(run_client())