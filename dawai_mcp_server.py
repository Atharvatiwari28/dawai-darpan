"""Dawai Darpan ka MCP server 🔌
Humare tools (tools.py) ko ek STANDARD plug pe lagata hai, taaki koi bhi AI app
(Claude Desktop, Cursor, kisi aur ka agent) unhe use kar sake.

Chalao (test):   python test_mcp.py
Claude Desktop:  claude_desktop_config.json mein is file ka path do
"""
import os
import sys
from pathlib import Path

# ⚠️ Claude Desktop is server ko KISI BHI folder se chala sakta hai.
#    Humara code "data/..." jaise relative path use karta hai, isliye pehle apne folder mein aao.
os.chdir(Path(__file__).resolve().parent)

from mcp.server.mcpserver import MCPServer
from tools import search_by_brand, search_by_salt, medicine_info, find_kendra

server = MCPServer(
    name="dawai-darpan",
    title="Dawai Darpan",
    description="Brand dawai ka saasta Jan Aushadhi generic, daam aur bachat.",
    # Yeh niyam har us AI ko milte hain jo is server se judta hai
    instructions=(
        "Use these tools to find cheaper Jan Aushadhi generic medicines in India. "
        "Only report prices and medicine info returned by the tools; never invent numbers. "
        "Always include the tool's disclaimer. Never give personal medical advice "
        "(which medicine to take, dose); tell the user to consult a doctor."
    ),
)

# Wahi functions, bas naya plug. Naam, docstring aur type hints se menu card apne-aap banta hai
server.add_tool(search_by_brand)
server.add_tool(search_by_salt)
server.add_tool(medicine_info)
server.add_tool(find_kendra)

if __name__ == "__main__":
    # stdio: Claude Desktop is program ko chalata hai aur stdin/stdout se baat karta hai.
    # ⚠️ Isliye yahan print() KABHI mat karna: stdout pe kuch bhi likha toh protocol toot jaayega.
    print("Dawai Darpan MCP server chalu (stdio)", file=sys.stderr)   # stderr theek hai
    server.run()
