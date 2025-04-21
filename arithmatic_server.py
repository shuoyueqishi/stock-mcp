

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

if __name__ == "__main__":
    mcp.run()
