This is a stock mcp base on open source akshare. It provides many tools for LLM to analyse stock or fund.
## Use it in Cline
```shell
git clone git@github.com:shuoyueqishi/stock-mcp.git

cd stock-mcp

uv async
```

configure cline MCP
```shell
"stock_mcp": {
      "autoApprove": [
      ],
      "disabled": false,
      "timeout": 60,
      "command": "uv",
      "args": [
        "--directory",
        "path\to\your\code",
        "run",
        "stock_server.py"
      ],
      "transportType": "stdio"
    }
```
