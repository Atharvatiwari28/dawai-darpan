"""MCP server ko ek asli MCP client ki tarah test karo (Claude Desktop jaisa)."""
import sys
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["dawai_mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                         # 🤝 haath milao

            tools = await session.list_tools()                 # 📜 menu card maango
            print("🔧 Server ke tools:")
            for t in tools.tools:
                print(f"   - {t.name}: {list(t.input_schema.get('properties', {}))}")

            result = await session.call_tool("search_by_brand", {"brand_name": "Dolo 650"})
            data = json.loads(result.content[0].text)
            print("\n📞 search_by_brand('Dolo 650') →")
            print(f"   {data['generic_name']}: ₹{data['generic_per_unit']} vs ₹{data['brand_per_unit']}"
                  f" ({data['saving_percent']}% bachat)")


asyncio.run(main())
