# Stock MCP 文档

基于开源akshare的股票和基金数据分析MCP服务器，为LLM提供丰富的金融数据分析工具。

## 功能概述

- 股票数据：获取个股历史行情、PE/PB估值数据
- 基金数据：查询基金净值、申购状态、持仓信息
- 市场指标：获取指数估值、基金仓位数据
- 新闻数据：个股新闻、新闻联播文字稿

## 工具目录

### 股票数据工具
- `get_single_stock_info`: 获取个股历史数据(含PE/ROE)
- `get_stock_indicator`: 获取指数成分股估值数据
- `get_stock_pe`: 获取市场PE历史数据
- `get_stock_pb`: 获取市场PB历史数据

### 基金数据工具
- `fund_purchase_em`: 查询基金申购状态
- `fund_info_index_em`: 获取指数基金信息
- `fund_portfolio_hold_em`: 查询基金持仓
- `fund_etf_spot_em`: 获取ETF实时行情

### 其他工具
- `stock_news_em`: 获取个股新闻
- `news_cctv`: 获取新闻联播文字稿

## 详细使用说明

### 股票数据示例

```python
# 获取贵州茅台历史数据
result = await get_single_stock_info(
    stock_code="600519",
    start_date="20240101",
    end_date="20240430",
    condition="pe < 40 AND roe > 20"
)
```

参数说明：
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| stock_code | str | 是 | 股票代码，如"600519" |
| start_date | str | 是 | 开始日期，格式"yyyyMMdd" |
| end_date | str | 是 | 结束日期，格式"yyyyMMdd" |
| condition | str | 否 | 筛选条件，如"pe < 20" |

### 基金数据示例

```python
# 查询沪深300指数基金
result = await fund_info_index_em(
    symbol="沪深指数",
    indicator="被动指数型",
    condition="近3年 > 10 AND 手续费 < 1.2"
)
```

## 最佳实践

1. 日期格式统一使用"yyyyMMdd"
2. 条件筛选支持复杂逻辑组合
3. 大数据量查询建议分时间段获取
4. 实时数据注意市场交易时间

## 安装配置

```shell
git clone git@github.com:shuoyueqishi/stock-mcp.git
cd stock-mcp
uv async
```

配置Cline MCP:
```json
"stock_mcp": {
  "autoApprove": [],
  "disabled": false,
  "timeout": 60,
  "command": "uv",
  "args": [
    "--directory",
    "path\\to\\your\\code",
    "run",
    "stock_server.py"
  ],
  "transportType": "stdio"
}
