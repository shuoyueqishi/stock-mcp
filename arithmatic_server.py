import asyncio
from datetime import datetime
import time
from fastmcp import FastMCP
mcp = FastMCP("Arithmatic MCP")

@mcp.tool()
async def add(a:float, b:float) -> float:
    """Add two numbers"""
    return a + b

@mcp.tool()
async def substract(a:float, b:float) -> float:
    """num a substract num b"""
    return a - b

@mcp.tool()
async def multiply(a:float, b:float) -> float:
    """num a multiplys num b"""
    return a * b

@mcp.tool()
async def divide(a:float, b:float) -> float:
    """num a divide num b"""
    return a / b

@mcp.tool()
async def get_current_time():
    """获取当前的日期时间
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# if __name__ == "__main__":
#     cur_time = asyncio.run(get_current_time())
#     print(cur_time)

if __name__ == "__main__":
    mcp.run()
