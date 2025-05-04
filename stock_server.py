import asyncio
import datetime
import operator
import re
import time
from typing import Dict, List, Union
from fastmcp import FastMCP
import pandas as pd
import akshare as ak
import numpy as np
from pydantic import Field

mcp = FastMCP("stock MCP", dependencies=["pandas", "numpy", "akshare"])

START_DATE = "20200101"
END_DATE = "20250418"


@mcp.tool()
async def get_single_stock_info(
    stock_code: str,
    start_date: str,
    end_date: str,
    retry: int = 3,
    condition: str = None,
):
    """
    根据股票代码获取股票的历史数据，包含PE、ROE等指标

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_code": {
          "type": "string",
          "description": "股票代码，如：'600000'",
          "examples": ["600000", "000001"]
        },
        "start_date": {
          "type": "string",
          "description": "开始日期，格式yyyyMMdd",
          "pattern": "^\\d{8}$",
          "examples": ["20200101", "20230101"]
        },
        "end_date": {
          "type": "string",
          "description": "结束日期，格式yyyyMMdd",
          "pattern": "^\\d{8}$",
          "examples": ["20201231", "20231231"]
        },
        "retry": {
          "type": "integer",
          "description": "请求失败后的重试次数",
          "default": 3,
          "minimum": 1,
          "maximum": 5
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "pe > 20 AND roe >= 0.15",
            "close > open AND volume > 100000"
          ]
        }
      },
      "required": ["stock_code", "start_date", "end_date"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量
        - code: 股票代码
        - transAmount: 成交额
        - amplitude: 振幅
        - riseFall: 涨跌幅(%)
        - riseFallAmount: 涨跌额
        - turnoverRate: 换手率(%)
        - pe: 滚动市盈率(TTM)
        - roe: 净资产收益率(%)

    Examples:
        ```python
        await get_single_stock_info(
            stock_code="600000",
            start_date="20230101",
            end_date="20231231",
            condition="pe < 15 AND roe > 0.1"
        )
        ```

    Error Handling:
        - 如果股票代码无效或数据获取失败，返回None
        - 如果条件语法错误，抛出ValueError
    """
    try:
        # 获取价格数据 (带重试)
        for _ in range(retry):
            try:
                # === 获取价格数据 ===
                price_df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    adjust="hfq",
                    start_date=start_date,
                    end_date=end_date,
                ).rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "股票代码": "code",
                        "成交额": "transAmount",
                        "振幅": "amplitude",
                        "涨跌幅": "riseFall",
                        "涨跌额": "riseFallAmount",
                        "换手率": "turnoverRate",
                    }
                )

                # 新浪财经接口（获取市盈率）
                pe_df = ak.stock_index_pe_lg()

                # 财报数据接口（获取ROE）
                finance_df = ak.stock_financial_abstract(symbol=stock_code)
                finance_df = finance_df[
                    finance_df["指标"].str.contains("ROE")
                    & finance_df["选项"].str.contains("盈利能力")
                ].copy()
                finance_df = finance_df.drop(columns=["选项"])
                report_date_cols = [
                    col for col in finance_df.columns if col.isdigit and len(col) == 8
                ]
                finance_df = pd.melt(
                    finance_df,
                    var_name="报表日期",
                    value_name="净资产收益率(ROE)",
                    value_vars=report_date_cols,
                )
                finance_df["date"] = pd.to_datetime(finance_df["报表日期"])
                finance_df["roe"] = finance_df["净资产收益率(ROE)"] / 100

                # === 数据合并 ===
                # 合并PE数据
                price_df["date"] = pd.to_datetime(price_df["date"])
                pe_df["日期"] = pd.to_datetime(pe_df["日期"])
                """
                市盈率指标：
                ╒══════════════════╤══════════════════════════╤═══════════════╤════════════╤═════════════════════════╕
                │ 指标名称         │ 公式                    │ 时间维度      │ 加权方式   │ 适用场景                │
                ╞══════════════════╪══════════════════════════╪═══════════════╪════════════╪═════════════════════════╡
                │ 静态市盈率       │ 总市值 / 上年净利润      │ 历史年度数据  │ 市值加权   │ 长期稳定盈利的公司      │
                ├──────────────────┼──────────────────────────┼───────────────┼────────────┼─────────────────────────┤
                │ 滚动市盈率(TTM)  │ 总市值 / 近4季度净利润   │ 最近12个月    │ 市值加权   │ 盈利波动大的行业        │
                ├──────────────────┼──────────────────────────┼───────────────┼────────────┼─────────────────────────┤
                │ 等权静态市盈率   │ 各股市值简单平均计算      │ 历史年度数据  │ 简单平均   │ 观察中小盘股估值        │
                ├──────────────────┼──────────────────────────┼───────────────┼────────────┼─────────────────────────┤
                │ 等权滚动市盈率   │ 各股市值简单平均计算      │ 最近12个月    │ 简单平均   │ 分析市场整体估值泡沫    │
                ╘══════════════════╧══════════════════════════╧═══════════════╧════════════╧═════════════════════════╛
                """
                merged_df = pd.merge_asof(
                    price_df.sort_values("date"),
                    pe_df[["日期", "等权滚动市盈率"]].rename(
                        columns={"日期": "date", "等权滚动市盈率": "pe"}
                    ),
                    on="date",
                    direction="backward",
                )

                # 合并ROE数据
                merged_df = pd.merge_asof(
                    merged_df.sort_values("date"),
                    finance_df[["date", "roe"]].sort_values("date"),
                    on="date",
                    direction="backward",
                )

                # === 数据清洗 ===
                merged_df["pe"] = merged_df["pe"].replace([np.inf, -np.inf], np.nan)
                merged_df["pe"] = merged_df["pe"].fillna(merged_df["pe"].median())
                merged_df["roe"] = merged_df["roe"].ffill()
                # 应用筛选条件
                if condition:
                    filtered_df = apply_filters_for_data_frame(merged_df, condition)
                    return filtered_df.to_markdown()
                return merged_df.head(10).to_markdown()
            except Exception as e:
                print(f"\n{stock_code}价格数据获取失败:{e}，重试中...")
                time.sleep(1)
    except Exception as e:
        print(f"{stock_code} 数据获取失败：{str(e)}\r\n")
        return None


@mcp.tool()
async def get_stock_indicator(
    stock_code: str, indicator_type: str = "沪深300", condition: str = None
):
    """
    根据股票代码获取指数历史行情数据，包含PE、PB和对应的百分位等

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_code": {
          "type": "string",
          "description": "股票代码，如：'600000'",
          "examples": ["600000", "000001"]
        },
        "indicator_type": {
          "type": "string",
          "description": "指数类型",
          "enum": ["上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180", "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"],
          "default": "沪深300"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "pe > 20 AND pb >= 0.5",
            "pe_percentile > 20 OR pb_percentile > 20"
          ]
        }
      },
      "required": ["stock_code"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - pe: 滚动市盈率(TTM)
        - pe_percentile: 市盈率历史百分位(近5年)
        - pb: 市净率
        - pb_percentile: 市净率历史百分位(近5年)

    Examples:
        ```python
        await get_stock_indicator(
            stock_code="600000",
            indicator_type="沪深300",
            condition="pe > 20 AND pb >= 0.5"
        )
        ```

    Error Handling:
        - 如果股票代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    # 获取基础行情
    df_price = ak.stock_zh_index_daily(stock_code)
    df_price["date"] = pd.to_datetime(df_price["date"])

    # 估值数据获取
    # 市盈率数据
    df_pe = _get_stock_pe(indicator_type)
    df_pe["date"] = pd.to_datetime(df_pe["date"])

    # 市净率数据
    df_pb = _get_stock_pb(indicator_type)

    df_pb["date"] = pd.to_datetime(df_pb["date"])

    # 融合PE
    merged_df = pd.merge(
        df_price, df_pe[["date", "pe_percentile"]], on="date", how="left"
    )
    merged_df["pe_percentile"] = merged_df["pe_percentile"].ffill().bfill()

    # 融合PB
    merged_df = pd.merge(
        merged_df, df_pb[["date", "pb_percentile"]], on="date", how="left"
    )
    merged_df["pb_percentile"] = merged_df["pb_percentile"].ffill().bfill()
    if condition:
        filtered_df = apply_filters_for_data_frame(merged_df, condition=condition)
        return filtered_df.to_markdown()
    return merged_df.to_markdown()


@mcp.tool()
async def get_stock_pe(stock_type: str = "沪深300", condition=None):
    """
    根据股票类型获取对应市场的历史PE数据(等权滚动市盈率)

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_type": {
          "type": "string",
          "description": "指数类型",
          "enum": ["上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180", "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"],
          "default": "沪深300"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "pe > 20 AND date >= '2024-01-01'",
            "pe_percentile > 20 OR pe <= 18"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - pe: 等权滚动市盈率(TTM)
        - pe_percentile: 市盈率历史百分位(近5年)

    Examples:
        ```python
        await get_stock_pe(
            stock_type="沪深300",
            condition="pe > 20 AND pe_percentile <= 80"
        )
        ```

    Error Handling:
        - 如果指数类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    pe_df = _get_stock_pe(stock_type)
    if condition:
        filter_df = apply_filters_for_data_frame(pe_df, condition)
        return filter_df.to_markdown()
    return pe_df.to_markdown()


def _get_stock_pe(stock_type: str = "沪深300"):
    """
    根据股票类型获取对应市场的历史PE数据(等权滚动市盈率)
    Args:
        stock_type: PE、PB历史指标类型，根据股票所处的分类区分，可选值有：上证50, 沪深300, 上证380, 创业板50, 中证500, 上证180, 深证红利, 深证100, 中证1000, 上证红利, 中证100, 中证800
    Returns: PE数据, 返回字段有：date、pe、pe_percentile
    """
    # 新版指数估值接口
    # 市盈率PE = 股票价格 / 每股（年度）盈利；
    # 较高的PE意味着市场认为该股票有更好的盈利增长潜力
    df = ak.stock_index_pe_lg(stock_type)

    # 数据清洗
    df = df[["日期", "等权滚动市盈率"]].rename(
        columns={"日期": "date", "等权滚动市盈率": "pe"}
    )
    df["date"] = pd.to_datetime(df["date"])

    # 前向填充法
    df["pe"] = df["pe"].fillna(method="ffill")

    # 异常值过滤，3σ 法则
    df = __filter_abnormal_3delta(df, "pe")
    df = __calculate_percentile(df, "pe", 1260)
    return df


@mcp.tool()
async def get_stock_pb(stock_type: str = "沪深300", condition: str = None):
    """
    根据股票类型获取对应市场的历史PB数据(等权市净率)

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_type": {
          "type": "string",
          "description": "指数类型",
          "enum": ["上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180", "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"],
          "default": "沪深300"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "pb < 1.5 AND pb_percentile <= 20",
            "date >= '2023-01-01'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - pb: 等权市净率
        - pb_percentile: 市净率历史百分位(近5年)

    Examples:
        ```python
        await get_stock_pb(
            stock_type="沪深300",
            condition="pb < 1.5 AND pb_percentile <= 20"
        )
        ```

    Error Handling:
        - 如果指数类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    pb_df = _get_stock_pb(stock_type)
    # 应用筛选条件
    if condition:
        return apply_filters_for_data_frame(pb_df, condition).to_markdown()
    return pb_df.to_markdown()


def _get_stock_pb(stock_type: str = "沪深300"):
    """
    根据股票类型获取对应市场的历史PB数据(等权市净率)

    Args:
        stock_type: 股票类型，可选值：
            "上证50", "沪深300", "上证380", "创业板50",
            "中证500", "上证180", "深证红利", "深证100",
            "中证1000", "上证红利", "中证100", "中证800"
    Returns:
        DataFrame包含字段：
        - date: 日期
        - pb: 等权市净率
        - pb_percentile: 历史百分位（近5年）

    使用示例：
    ```python
    _get_stock_pb("沪深300")
    ```
    """
    # 市净率PB = 股票价格 / 每股净资产
    # 用于衡量投资者为获得每一元净资产愿意支付多少元股价，较低的PB意味着市场低估了该公司的净资产
    pb_df = ak.stock_index_pb_lg(stock_type)

    # 数据清洗
    pb_df = pb_df[["日期", "等权市净率"]].rename(
        columns={"日期": "date", "等权市净率": "pb"}
    )
    pb_df["date"] = pd.to_datetime(pb_df["date"])

    # 前向填充法
    pb_df["pb"] = pb_df["pb"].fillna(method="ffill")

    # 异常值过滤，3σ 法则
    pb_df = __filter_abnormal_3delta(pb_df, "pb")
    pb_df = __calculate_percentile(pb_df, "pb", 1260)
    return pb_df


@mcp.tool()
async def fund_purchase_em(condition: str = None):
    """
    获取所有基金申购状态

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "基金代码 = 'SZ60606' AND 手续费 <= 3%",
            "基金类型 = '指数型'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 基金类型: 基金类型
        - 最新净值/万份收益: 最新净值或万份收益
        - 最新净值/万份收益-报告时间: 数据报告时间
        - 申购状态: 申购状态(开放/暂停)
        - 赎回状态: 赎回状态(开放/暂停)
        - 下一开放日: 下一开放申购日期
        - 购买起点: 最低购买金额
        - 日累计限定金额: 单日累计限额
        - 手续费: 申购费率(%)

    Examples:
        ```python
        await fund_purchase_em(
            condition="基金代码 = 'SZ60606' AND 手续费 <= 3%"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_purchase_em()
    if condition:
        return apply_filters_for_data_frame(df, condition=condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_name_em():
    """
    获取东方财富网站-天天基金网-基金数据-所有基金的名称和类型
    数据来源：https://fund.eastmoney.com/manager/default.html#dt14;mcreturnjson;ftall;pn20;pi1;scabbname;stasc
    Returns:
      所有基金的名称和类型
    """
    df = ak.fund_name_em()
    return df.to_markdown()


@mcp.tool()
async def fund_info_index_em(
    symbol: str = "沪深指数", indicator: str = "被动指数型", condition: str = None
):
    """
    获取指数型基金近几年的单位净值、增长率、手续费等

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "行业类型",
          "enum": ["全部", "沪深指数", "行业主题", "大盘指数", "中盘指数", "小盘指数", "股票指数", "债券指数"],
          "default": "沪深指数"
        },
        "indicator": {
          "type": "string",
          "description": "基金类型",
          "enum": ["全部", "被动指数型", "增强指数型"],
          "default": "被动指数型"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "近3年 > 5 AND 近2年 > 5",
            "日期 >= '2023-01-01'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 基金代码: 基金唯一标识
        - 基金名称: 基金全称
        - 单位净值: 最新单位净值
        - 日期: 净值日期
        - 日增长率: 单日增长率(%)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 手续费: 申购费率(%)
        - 起购金额: 最低申购金额(元)

    Examples:
        ```python
        await fund_info_index_em(
            symbol="沪深指数",
            indicator="被动指数型",
            condition="近3年 > 5 AND 手续费 < 1.5"
        )
        ```

    Error Handling:
        - 如果参数类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_info_index_em(symbol, indicator)
    df.sort_values("近1周", axis=1, ascending=False)
    if condition:
        return apply_filters_for_data_frame(df, condition=condition).head(10).to_markdown()
    return df.head(10).to_markdown()


@mcp.tool()
async def fund_open_fund_info_em(
    symbol: str = "710001",
    indicator: str = "单位净值走势",
    period: str = "成立来",
    condition: str = None,
):
    """
    获取指定基金特定指标的数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["710001", "000001"]
        },
        "indicator": {
          "type": "string",
          "description": "需要获取的指标",
          "enum": ["单位净值走势", "累计净值走势", "累计收益率走势", "同类排名走势", "分红送配详情", "拆分详情"],
          "default": "单位净值走势"
        },
        "period": {
          "type": "string",
          "description": "统计期间",
          "enum": ["1月", "3月", "6月", "1年", "3年", "5年", "今年来", "成立来"],
          "default": "成立来"
        }
      },
      "required": ["symbol"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 日期: 数据日期
        - 单位净值: 每份基金单位的净值
        - 日增长率: 单日增长率(%)
        - 其他指标: 根据indicator参数返回相应指标数据

    Examples:
        ```python
        await fund_open_fund_info_em(
            symbol="710001",
            indicator="单位净值走势",
            period="1年"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果指标类型无效，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_open_fund_info_em(symbol, indicator, period)
    df.sort_values("日增长率", axis=1, ascending=False)
    if condition:
        return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
    return df.head(10).to_markdown()


@mcp.tool()
async def fund_money_fund_daily_em(condition: str = None):
    """
    获取当前交易日的所有货币型基金收益数据列表

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）。示例：'手续费 < 1.5 AND 上一个交易日-7日年化% >= 2.3'",
          "examples": [
            "手续费 < 1.5 AND 日涨幅 <= 1",
            "上一个交易日-万份收益 >= 1.5"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 基金代码
        - 基金简称
        - 上一个交易日-万份收益
        - 上一个交易日-7日年化%
        - 上一个交易日-单位净值
        - 上两个交易日-万份收益
        - 上两个交易日-7日年化%
        - 上两个交易日-单位净值
        - 日涨幅
        - 成立日期
        - 基金经理
        - 手续费
        - 可购全部

    Examples:
        ```python
        await fund_money_fund_daily_em(
            condition="手续费 < 1.5 AND 上一个交易日-7日年化% >= 2.3"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_money_fund_daily_em()
    _rename_dynamic_date_columns(df, "万份收益")
    _rename_dynamic_date_columns(df, "7日年化%")
    _rename_dynamic_date_columns(df, "单位净值")

    # 处理7日年化%列的值，去除%
    for col in df.columns:
        df[col] = df[col].replace("---", "0")
        df[col] = df[col].replace("0费率", "0")
        if "7日年化" in col:
            df[col] = df[col].astype(str).str.replace("%", "").astype(float)

    df.sort_values("上一个交易日-7日年化%", axis=1, ascending=False)
    if condition:
        return (
            apply_filters_for_data_frame(df, condition=condition).head(10).to_markdown()
        )
    return df.head(10).to_markdown()


def _rename_dynamic_date_columns(df: pd.DataFrame, suffix: str = None):
    # 获取所有包含日期的列名
    date_start_cols = [
        col for col in df.columns if re.match(r"\d{4}-\d{2}-\d{2}-(.*)", col)
    ]

    target_cols = [col for col in date_start_cols if col.rfind(suffix) != -1]

    # 提取日期并排序
    dates = sorted(
        [re.search(r"(\d{4}-\d{2}-\d{2})", col).group(1) for col in target_cols],
        reverse=True,
    )

    # 只处理最近两个交易日
    if len(dates) >= 1:
        df.columns = df.columns.str.replace(dates[0], "上一个交易日")
    if len(dates) >= 2:
        df.columns = df.columns.str.replace(dates[1], "上两个交易日")


@mcp.tool()
async def fund_money_fund_info_em(symbol: str = "000009", condition: str = None):
    """
    获取指定的货币型基金收益-历史净值数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "货币型基金代码",
          "examples": ["000009", "000001"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "万份收益 >= 1.5",
            "7日年化% > 2.5"
          ]
        }
      },
      "required": ["symbol"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 日期: 数据日期
        - 万份收益: 每万份基金收益(元)
        - 7日年化%: 7日年化收益率(%)
        - 申购状态: 开放/暂停
        - 赎回状态: 开放/暂停

    Examples:
        ```python
        await fund_money_fund_info_em(
            symbol="000009",
            condition="万份收益 >= 1.5 AND 7日年化% > 2.5"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_money_fund_info_em(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
    return df.head(10).to_markdown()


@mcp.tool()
async def fund_etf_fund_daily_em(condition: str = None):
    """
    获取所有场内基金数据列表

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "折价率 < 1 AND 增长率 > 0",
            "类型 = 'ETF'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 基金代码: 基金唯一标识
        - 基金简称: 基金简称
        - 类型: 基金类型(ETF/LOF等)
        - 前一个交易日-单位净值: 单位净值(元)
        - 前一个交易日-累计净值: 累计净值(元)
        - 前两个交易日-单位净值: 单位净值(元)
        - 前两个交易日-累计净值: 累计净值(元)
        - 增长值: 净值增长值(元)
        - 增长率: 净值增长率(%)
        - 市价: 市场价格(元)
        - 折价率: 折溢价率(%)

    Examples:
        ```python
        await fund_etf_fund_daily_em(
            condition="折价率 < 1 AND 增长率 > 0"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_etf_fund_daily_em()
    _rename_dynamic_date_columns(df, "单位净值")
    _rename_dynamic_date_columns(df, "累计净值")
    df.sort_values("增长率", axis=1, ascending=False)
    if condition:
        return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
    return df.head(10).to_markdown()


@mcp.tool()
async def fund_etf_fund_info_em(
    fund: str = "511280",
    start_date: str = "20000101",
    end_date: str = "20500101",
    condition: str = None,
):
    """
    获取指定的场内交易基金的历史净值明细

    Input Schema:
    {
      "type": "object",
      "properties": {
        "fund": {
          "type": "string",
          "description": "场内交易基金代码",
          "examples": ["511280", "510300"]
        },
        "start_date": {
          "type": "string",
          "description": "开始日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20200101", "20230101"]
        },
        "end_date": {
          "type": "string",
          "description": "结束日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20201231", "20231231"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "单位净值 > 1.5 AND 日增长率 > 0",
            "净值日期 >= '2023-01-01'"
          ]
        }
      },
      "required": ["fund"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 净值日期: 净值公布日期
        - 单位净值: 每份基金单位的净值
        - 累计净值: 累计单位净值
        - 日增长率: 单日增长率(%)
        - 申购状态: 开放/暂停
        - 赎回状态: 开放/暂停

    Examples:
        ```python
        await fund_etf_fund_info_em(
            fund="511280",
            start_date="20230101",
            end_date="20231231",
            condition="单位净值 > 1.5 AND 日增长率 > 0"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果日期格式错误，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_etf_fund_info_em(fund, start_date, end_date)
    if condition:
        return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
    return df.head(10).to_markdown()


@mcp.tool()
async def fund_value_estimation_em(symbol: str = "全部", condition: str = None):
    """
    按照类型获取近期净值估算数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金类型",
          "enum": ["全部", "股票型", "混合型", "债券型", "指数型", "QDII", "ETF联接", "LOF", "场内交易基金"],
          "default": "全部"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "估算偏差 > 0.5",
            "基金名称包含('沪深300')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 基金代码: 基金唯一标识
        - 基金名称: 基金全称
        - 估算偏差: 净值估算与实际净值的偏差(%)

    Examples:
        ```python
        await fund_value_estimation_em(
            symbol="股票型",
            condition="估算偏差 > 0.5"
        )
        ```

    Error Handling:
        - 如果类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_value_estimation_em(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_aum_em(condition: str = None):
    """
    获取基金公司排名列表

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "全部管理规模 > 1000",
            "基金公司 = '易方达'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金公司: 基金公司名称
        - 成立时间: 公司成立日期
        - 全部管理规模: 管理总规模(亿元)
        - 全部基金数: 旗下基金总数
        - 全部经理数: 基金经理人数

    Examples:
        ```python
        await fund_aum_em(
            condition="全部管理规模 > 1000 AND 基金公司包含('易方达')"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_aum_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_aum_trend_em(condition: str = None) -> pd.DataFrame:
    """
    基金市场管理规模走势图

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "value > 10000",
            "date >= '2023-01-01'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - value: 基金市场管理规模(亿元)

    Examples:
        ```python
        await fund_aum_trend_em(
            condition="value > 10000 AND date >= '2023-01-01'"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_aum_trend_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_aum_hist_em(year: str = "2023", condition: str = None) -> pd.DataFrame:
    """
    获取基金公司历年管理规模排行列表

    Input Schema:
    {
      "type": "object",
      "properties": {
        "year": {
          "type": "string",
          "description": "年份(yyyy格式)",
          "pattern": "^\\d{4}$",
          "examples": ["2023", "2024"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "总规模 > 1000 AND 股票型 > 500",
            "基金公司 = '易方达'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金公司: 基金公司名称
        - 总规模: 管理总规模(亿元)
        - 股票型: 股票型基金规模(亿元)
        - 混合型: 混合型基金规模(亿元)
        - 债券型: 债券型基金规模(亿元)
        - 指数型: 指数型基金规模(亿元)
        - QDII: QDII基金规模(亿元)
        - 货币型: 货币型基金规模(亿元)

    Examples:
        ```python
        await fund_aum_hist_em(
            year="2023",
            condition="总规模 > 1000 AND 股票型 > 500"
        )
        ```

    Error Handling:
        - 如果年份格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_aum_hist_em(year)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_announcement_personnel_em(
    symbol: str = "000001", condition: str = None
) -> pd.DataFrame:
    """
    基金的人事调整-公告列表
    Args:
        symbol: 基金代码; 可以通过调用 ak.fund_name_em() 接口获取
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "基金代码 = '000001'"
    Returns:  基金的人事调整-公告列表，返回字段：
                - 基金代码
                - 公告标题
                - 基金名称
                - 公告日期
                - 报告ID
    """
    df = ak.fund_announcement_personnel_em(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_spot_em(condition: str = None) -> pd.DataFrame:
    """
    ETF 实时行情

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "换手率 > 20 AND 涨跌额 >= 10",
            "成交量 > 100000 AND 涨跌幅 < 5"
          ]
        }
      },
      "required": []
    }

    Args:
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "换手率 > 20 AND 涨跌额 >= 10"
    Returns: ETF 实时行情列表，返回的字段有：
        - 代码
        - 名称
        - 最新价
        - IOPV实时估值
        - 基金折价率
        - 涨跌额
        - 涨跌幅
        - 成交量
        - 成交额
        - 开盘价
        - 最高价
        - 最低价
        - 昨收
        - 振幅
        - 换手率
        - 量比
        - 委比
        - 外盘
        - 内盘
        - 主力净流入-净额
        - 主力净流入-净占比
        - 超大单净流入-净额
        - 超大单净流入-净占比
        - 大单净流入-净额
        - 大单净流入-净占比
        - 中单净流入-净额
        - 中单净流入-净占比
        - 小单净流入-净额
        - 小单净流入-净占比
        - 现手
        - 买一
        - 卖一
        - 最新份额
        - 流通市值
        - 总市值
        - 数据日期
        - 更新时间

    Examples:
        ```python
        await fund_etf_spot_em(
            condition="换手率 > 20 AND 涨跌额 >= 10"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_spot_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_hist_em(
    symbol: str = "159707",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
    condition: str = None,
) -> pd.DataFrame:
    """
    获取特定基金的每日ETF行情数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "ETF代码",
          "examples": ["159707", "510300"]
        },
        "period": {
          "type": "string",
          "description": "时间周期",
          "enum": ["daily", "weekly", "monthly"],
          "default": "daily"
        },
        "start_date": {
          "type": "string",
          "description": "开始日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20200101", "20230101"]
        },
        "end_date": {
          "type": "string",
          "description": "结束日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20201231", "20231231"]
        },
        "adjust": {
          "type": "string",
          "description": "复权方式",
          "enum": ["", "qfq", "hfq"],
          "default": ""
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > open AND volume > 100000",
            "date >= '2023-01-01'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: ETF 代码
        period: 类型，选项有：{'daily', 'weekly', 'monthly'}
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式，选项有：{"qfq": "前复权", "hfq": "后复权", "": "不复权"}
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "收盘 > open AND 成交量 > 100000"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 日期
        - 开盘
        - 收盘
        - 最高
        - 最低
        - 成交量
        - 成交额
        - 振幅
        - 涨跌幅
        - 涨跌额
        - 换手率

    Examples:
        ```python
        await fund_etf_hist_em(
            symbol="159707",
            condition="收盘 > open AND 成交量 > 100000"
        )
        ```

    Error Handling:
        - 如果ETF代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_hist_em(symbol, period, start_date, end_date, adjust)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_hist_min_em(
    symbol: str = "159707",
    start_date: str = "1979-09-01 09:32:00",
    end_date: str = "2222-01-01 09:32:00",
    period: str = "5",
    adjust: str = "",
    condition: str = None,
) -> pd.DataFrame:
    """
    获取特定基金的时分ETF行情数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "ETF代码",
          "examples": ["159707", "510300"]
        },
        "start_date": {
          "type": "string",
          "description": "开始日期(yyyy-MM-dd HH:mm:ss)",
          "examples": ["2023-01-01 09:30:00"]
        },
        "end_date": {
          "type": "string",
          "description": "结束日期(yyyy-MM-dd HH:mm:ss)",
          "examples": ["2023-12-31 15:00:00"]
        },
        "period": {
          "type": "string",
          "description": "时间间隔(分钟)",
          "enum": ["1", "5", "15", "30", "60"],
          "default": "5"
        },
        "adjust": {
          "type": "string",
          "description": "复权方式",
          "enum": ["", "qfq", "hfq"],
          "default": ""
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > open AND volume > 10000",
            "time >= '09:30:00' AND time <= '11:30:00'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: ETF 代码
        start_date: 开始日期
        end_date: 结束日期
        period: 时间间隔类型，选项：{"1", "5", "15", "30", "60"}
        adjust: 复权方式，选项：{'', 'qfq', 'hfq'}
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "close > open AND volume > 10000"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 时间
        - 开盘
        - 收盘
        - 最高
        - 最低
        - 涨跌幅
        - 涨跌额
        - 成交量
        - 成交额
        - 振幅
        - 换手率

    Examples:
        ```python
        await fund_etf_hist_min_em(
            symbol="159707",
            condition="close > open AND volume > 10000"
        )
        ```

    Error Handling:
        - 如果ETF代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_hist_min_em(symbol, period, start_date, end_date, adjust)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_fee_em(
    symbol: str = "015641",
    indicator: str = "认购费率",
) -> pd.DataFrame:
    """
    基金的交易规则，费率等

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["015641", "000001"]
        },
        "indicator": {
          "type": "string",
          "description": "指标类型",
          "enum": ["交易状态", "申购与赎回金额", "交易确认日", "运作费用", "认购费率", "申购费率", "赎回费率"],
          "default": "认购费率"
        },
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金代码
        indicator: 指标，可选值：{"交易状态", "申购与赎回金额", "交易确认日", "运作费用", "认购费率", "申购费率", "赎回费率"}
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 指标名称
        - 费率(%)
        - 其他相关字段(根据指标类型变化)

    Examples:
        ```python
        await fund_fee_em(
            symbol="015641",
            indicator="认购费率",
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_fee_em(symbol, indicator)
    return df.to_markdown()


@mcp.tool()
async def fund_etf_category_sina(
    symbol: str = "LOF基金", condition: str = None
) -> pd.DataFrame:
    """
    指定类型的基金列表

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金类型",
          "enum": ["封闭式基金", "ETF基金", "LOF基金"],
          "default": "LOF基金"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "基金代码 = 'sh510050'",
            "基金名称包含('沪深300')"
          ]
        }
      },
      "required": []
    }

    Args:
        symbol: 类型，可选值有："封闭式基金", "ETF基金", "LOF基金"
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "基金代码 = 'sh510050'"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 代码
        - 名称
        - 最新价
        - 涨跌额
        - 涨跌幅
        - 买入
        - 卖出
        - 昨收
        - 今开
        - 最高
        - 最低
        - 成交量
        - 成交额

    Examples:
        ```python
        await fund_etf_category_sina(
            symbol="ETF基金",
            condition="基金名称包含('沪深300')"
        )
        ```

    Error Handling:
        - 如果类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_category_sina(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_hist_sina(
    symbol: str = "sh510050", condition: str = None
) -> pd.DataFrame:
    """
    基金的日行情数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["sh510050", "sz159915"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > open AND volume > 100000",
            "date >= '2023-01-01'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金名称, 可以通过 ak.fund_etf_category_sina() 函数获取
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "close > open AND volume > 100000"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量

    Examples:
        ```python
        await fund_etf_hist_sina(
            symbol="sh510050",
            condition="close > open AND volume > 100000"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_hist_sina(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_dividend_sina(
    symbol: str = "sh510050", condition: str = None
) -> pd.DataFrame:
    """
    基金的累计分红数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["sh510050", "sz159915"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "累计分红 > 0.5",
            "日期 = '2023-01-01'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金名称, 可以通过 ak.fund_etf_category_sina() 函数获取
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "分红金额 > 0.5"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 日期
        - 累计分红

    Examples:
        ```python
        await fund_etf_dividend_sina(
            symbol="sh510050",
            condition="累计分红 > 0.5"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_dividend_sina(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_spot_ths(date: str = "", condition: str = None) -> pd.DataFrame:
    """
    同花顺理财-基金数据-每日净值-ETF-实时行情

    Input Schema:
    {
      "type": "object",
      "properties": {
        "date": {
          "type": "string",
          "description": "交易日期(yyyy-MM-dd)",
          "examples": ["2023-01-01", "2024-05-01"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "涨跌幅 > 1 AND 成交量 > 100000",
            "基金代码 = '510300'"
          ]
        }
      },
      "required": []
    }

    Args:
        date: 交易日期
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "涨跌幅 > 1 AND 成交量 > 100000"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号
        - 基金代码
        - 基金名称
        - 当前-单位净值
        - 当前-累计净值
        - 前一日-单位净值
        - 前一日-累计净值
        - 增长值
        - 增长率
        - 赎回状态
        - 申购状态
        - 最新-交易日
        - 最新-单位净值
        - 最新-累计净值
        - 基金类型

    Examples:
        ```python
        await fund_etf_spot_ths(
            date="2024-05-01",
            condition="涨跌幅 > 1 AND 成交量 > 100000"
        )
        ```

    Error Handling:
        - 如果日期格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_spot_ths(date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_individual_basic_info_xq(
    symbol: str = "000001", timeout: float = None, condition: str = None
) -> pd.DataFrame:
    """
    基金基本信息

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "timeout": {
          "type": "number",
          "description": "超时时间(秒)",
          "default": null
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "基金类型 = '股票型'",
            "成立日期 >= '2020-01-01'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金代码
        timeout: 超时时间(秒)
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "基金类型 = '股票型'"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 基金代码
        - 基金名称
        - 基金全称
        - 成立时间
        - 最新规模
        - 基金公司
        - 基金经理
        - 托管银行
        - 基金类型
        - 评级机构
        - 基金评级
        - 投资策略
        - 投资目标
        - 业绩比较基准

    Examples:
        ```python
        await fund_individual_basic_info_xq(
            symbol="000001",
            condition="基金类型 = '股票型'"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_individual_basic_info_xq(symbol, timeout)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_individual_achievement_xq(
    symbol: str = "000001", timeout: float = None, condition: str = None
) -> pd.DataFrame:
    """
    基金业绩

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "timeout": {
          "type": "number",
          "description": "超时时间(秒)",
          "default": null
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "近1年收益率 > 10",
            "成立以来年化 >= 8"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金代码
        timeout: 超时时间(秒)
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "近1年收益率 > 10"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 业绩类型
        - 周期
        - 本产品区间收益
        - 本产品最大回撒
        - 周期收益同类排名

    Examples:
        ```python
        await fund_individual_achievement_xq(
            symbol="000001",
            condition="近1年收益率 > 10"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_individual_achievement_xq(symbol, timeout)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_individual_analysis_xq(
    symbol: str = "000001", timeout: float = None, condition: str = None
) -> pd.DataFrame:
    """
    基金数据分析

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "timeout": {
          "type": "number",
          "description": "超时时间(秒)",
          "default": null
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件",
          "examples": [
            "年化夏普比率 > 1",
            "最大回撤 < 20"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金代码
        timeout: 超时时间(秒)
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "夏普比率 > 1"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 周期
        - 较同类风险收益比
        - 较同类抗风险波动
        - 年化波动率
        - 年化夏普比率
        - 最大回撤

    Examples:
        ```python
        await fund_individual_analysis_xq(
            symbol="000001",
            condition="年化夏普比率 > 1"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_individual_analysis_xq(symbol, timeout)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_individual_profit_probability_xq(
    symbol: str = "000001", timeout: float = None, condition: str = None
) -> pd.DataFrame:
    """
    雪球基金-盈利概率-历史任意时点买入，持有满 X 年，盈利概率 Y%

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "timeout": {
          "type": "number",
          "description": "超时时间(秒)",
          "default": null
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "持有时长 > 70",
            "盈利概率 >= 90"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金代码
        timeout: 超时时间(秒)
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "持有1年盈利概率 > 70"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 持有时长
        - 盈利概率
        - 平均收益

    Examples:
        ```python
        await fund_individual_profit_probability_xq(
            symbol="000001",
            condition="盈利概率 > 70"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_individual_profit_probability_xq(symbol, timeout)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_individual_detail_info_xq(
    symbol: str = "000001", timeout: float = None, condition: str = None
) -> pd.DataFrame:
    """
    雪球基金-交易规则

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "timeout": {
          "type": "number",
          "description": "超时时间(秒)",
          "default": null
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "条件或名称 < 1.5",
            "赎回费率 = 0"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金代码
        timeout: 超时时间(秒)
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "费用类型 = '管理费'"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 费用类型
        - 条件或名称
        - 费用

    Examples:
        ```python
        await fund_individual_detail_info_xq(
            symbol="000001",
            condition="费用类型 = '管理费'"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_individual_detail_info_xq(symbol, timeout)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_individual_detail_hold_xq(
    symbol: str = "002804",
    date: str = "20231231",
    timeout: float = None,
    condition: str = None,
) -> pd.DataFrame:
    """
    雪球基金-持仓

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["002804", "000001"]
        },
        "date": {
          "type": "string",
          "description": "财报日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20231231", "20230630"]
        },
        "timeout": {
          "type": "number",
          "description": "超时时间(秒)",
          "default": null
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "持仓比例 > 5"
          ]
        }
      },
      "required": ["symbol", "date"]
    }

    Args:
        symbol: 基金代码
        date: 财报日期
        timeout: 超时时间(秒)
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "持仓比例 > 5"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 资产类型
        - 仓位占比

    Examples:
        ```python
        await fund_individual_detail_hold_xq(
            symbol="002804",
            date="20231231",
            condition="仓位占比 > 5"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果日期格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_individual_detail_hold_xq(symbol, date, timeout)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_manager_em(condition: str = None) -> pd.DataFrame:
    """
    天天基金网-基金数据-基金经理大全

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "现任基金资产总规模 > 1000",
            "姓名包含('张')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 姓名: 基金经理姓名
        - 所属公司: 基金公司名称
        - 现任基金: 当前管理的基金列表
        - 累计从业时间: 从业年限(年)
        - 现任基金资产总规模: 管理总规模(亿元)
        - 现任基金最佳回报: 最佳基金回报率(%)

    Examples:
        ```python
        await fund_manager_em(
            condition="现任基金资产总规模 > 1000 AND 累计从业时间 >= 5"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_manager_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_portfolio_hold_em(
    symbol: str = "000001", date: str = "2024", condition: str = None
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-基金持仓

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "date": {
          "type": "string",
          "description": "查询年份(yyyy格式)",
          "pattern": "^\\d{4}$",
          "examples": ["2023", "2024"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "占净值比例 > 5 AND 持仓市值 > 100000",
            "股票代码 = '600000'"
          ]
        }
      },
      "required": ["symbol", "date"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 股票代码: 股票代码
        - 股票名称: 股票名称
        - 占净值比例: 占基金净值比例(%)
        - 持股数: 持股数量(股)
        - 持仓市值: 持仓市值(元)
        - 季度: 报告季度

    Examples:
        ```python
        await fund_portfolio_hold_em(
            symbol="000001",
            date="2024",
            condition="占净值比例 > 5 AND 持仓市值 > 100000"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果年份格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_portfolio_hold_em(symbol, date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_portfolio_bond_hold_em(
    symbol: str = "000001", date: str = "2023", condition: str = None
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-债券持仓

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "date": {
          "type": "string",
          "description": "查询年份(yyyy格式)",
          "pattern": "^\\d{4}$",
          "examples": ["2023", "2024"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "占净值比例 > 5 AND 持仓市值 > 100000",
            "债券代码 = '123456'"
          ]
        }
      },
      "required": ["symbol", "date"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 债券代码: 债券代码
        - 债券名称: 债券名称
        - 占净值比例: 占基金净值比例(%)
        - 持仓市值: 持仓市值(元)
        - 季度: 报告季度

    Examples:
        ```python
        await fund_portfolio_bond_hold_em(
            symbol="000001",
            date="2023",
            condition="占净值比例 > 5 AND 持仓市值 > 100000"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果年份格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_portfolio_bond_hold_em(symbol, date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_portfolio_industry_allocation_em(
    symbol: str = "000001", date: str = "2023", condition: str = None
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-行业配置

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["000001", "002804"]
        },
        "date": {
          "type": "string",
          "description": "查询年份(yyyy格式)",
          "pattern": "^\\d{4}$",
          "examples": ["2023", "2024"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "占净值比例 > 5 AND 市值 > 100000",
            "行业类别 = '制造业'"
          ]
        }
      },
      "required": ["symbol", "date"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 行业类别: 行业分类名称
        - 占净值比例: 占基金净值比例(%)
        - 市值: 持仓市值(元)
        - 截止时间: 数据截止日期

    Examples:
        ```python
        await fund_portfolio_industry_allocation_em(
            symbol="000001",
            date="2023",
            condition="占净值比例 > 5 AND 市值 > 100000"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果年份格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_portfolio_industry_allocation_em(symbol, date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_portfolio_change_em(
    symbol: str = "003567",
    indicator: str = "累计买入",
    date: str = "2023",
    condition: str = None,
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-重大变动

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["003567", "000001"]
        },
        "indicator": {
          "type": "string",
          "description": "指标类型",
          "enum": ["累计买入", "累计卖出"],
          "default": "累计买入"
        },
        "date": {
          "type": "string",
          "description": "查询年份(yyyy格式)",
          "pattern": "^\\d{4}$",
          "examples": ["2023", "2024"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "本期累计买入金额 > 10000000",
            "股票代码 = '600000'"
          ]
        }
      },
      "required": ["symbol", "indicator", "date"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 股票代码: 股票代码
        - 股票名称: 股票名称
        - 本期累计买入金额: 累计买入金额(元)
        - 占期初基金资产净值比例: 占净值比例(%)
        - 季度: 报告季度

    Examples:
        ```python
        await fund_portfolio_change_em(
            symbol="003567",
            indicator="累计买入",
            date="2023",
            condition="本期累计买入金额 > 10000000"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果指标类型无效，抛出ValueError
        - 如果年份格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_portfolio_change_em(symbol, indicator, date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_stock_position_lg(condition: str = None) -> pd.DataFrame:
    """
    乐咕乐股-基金仓位-股票型基金仓位

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "position > 60 AND date >= '2024-01-01'",
            "close > position"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - close: 收盘点位
        - position: 股票仓位(%)

    Examples:
        ```python
        await fund_stock_position_lg(
            condition="position > 60 AND date >= '2024-01-01'"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_stock_position_lg()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_balance_position_lg(condition: str = None) -> pd.DataFrame:
    """
    乐咕乐股-基金仓位-平衡混合型基金仓位

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "position > 50 AND date >= '2024-01-01'",
            "close > position"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - close: 收盘点位
        - position: 平衡混合型基金仓位(%)

    Examples:
        ```python
        await fund_balance_position_lg(
            condition="position > 50 AND date >= '2024-01-01'"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_balance_position_lg()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_linghuo_position_lg(condition: str = None) -> pd.DataFrame:
    """
    乐咕乐股-基金仓位-灵活配置型基金仓位

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "position > 40 AND date >= '2024-01-01'",
            "close > position"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - close: 收盘点位
        - position: 灵活配置型基金仓位(%)

    Examples:
        ```python
        await fund_linghuo_position_lg(
            condition="position > 40 AND date >= '2024-01-01'"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_linghuo_position_lg()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_open_fund_rank_em(
    symbol: str = "全部", condition: str = None
) -> pd.DataFrame:
    """
    东方财富网-数据中心-开放基金排行

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金类型",
          "enum": ["全部", "股票型", "混合型", "债券型", "指数型", "QDII", "FOF"],
          "default": "全部"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "近1年 > 10 AND 手续费 < 1.5",
            "基金简称包含('沪深300')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 日期: 净值日期
        - 单位净值: 单位净值(元)
        - 累计净值: 累计净值(元)
        - 日增长率: 日增长率(%)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 自定义: 自定义收益率(%)
        - 手续费: 申购费率(%)

    Examples:
        ```python
        await fund_open_fund_rank_em(
            symbol="股票型",
            condition="近1年 > 10 AND 手续费 < 1.5"
        )
        ```

    Error Handling:
        - 如果基金类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_open_fund_rank_em(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
    return df.head(10).to_markdown()


@mcp.tool()
async def fund_exchange_rank_em(condition: str = None) -> pd.DataFrame:
    """
    东方财富网-数据中心-场内交易基金排行

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "近1年 > 10 AND 类型 = 'ETF'",
            "成立日期 >= '2020-01-01'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 类型: 基金类型(ETF/LOF等)
        - 日期: 净值日期
        - 单位净值: 单位净值(元)
        - 累计净值: 累计净值(元)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 成立日期: 基金成立日期

    Examples:
        ```python
        await fund_exchange_rank_em(
            condition="近1年 > 10 AND 类型 = 'ETF'"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_exchange_rank_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_money_rank_em(condition: str = None) -> pd.DataFrame:
    """
    东方财富网-数据中心-货币型基金排行

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "年化收益率7日 > 2.5 AND 手续费 = 0",
            "基金简称包含('宝')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 日期: 净值日期
        - 万份收益: 每万份收益(元)
        - 年化收益率7日: 7日年化收益率(%)
        - 年化收益率14日: 14日年化收益率(%)
        - 年化收益率28日: 28日年化收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 近5年: 近5年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 手续费: 申购费率(%)

    Examples:
        ```python
        await fund_money_rank_em(
            condition="年化收益率7日 > 2.5 AND 手续费 = 0"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_money_rank_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_lcx_rank_em(condition: str = None) -> pd.DataFrame:
    """
    东方财富网-数据中心-理财基金排行

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "年化收益率-7日 > 2.5 AND 手续费 = 0",
            "基金简称包含('理财')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 日期: 净值日期
        - 万份收益: 每万份收益(元)
        - 年化收益率-7日: 7日年化收益率(%)
        - 年化收益率-14日: 14日年化收益率(%)
        - 年化收益率-28日: 28日年化收益率(%)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 可购买: 是否可购买
        - 手续费: 申购费率(%)

    Examples:
        ```python
        await fund_lcx_rank_em(
            condition="年化收益率-7日 > 2.5 AND 手续费 = 0"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_lcx_rank_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_hk_rank_em(condition: str = None) -> pd.DataFrame:
    """
    东方财富网-数据中心-香港基金排行

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "日增长率 > 0 AND 近1年 > 10",
            "基金简称包含('中国')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 币种: 计价币种
        - 日期: 净值日期
        - 单位净值: 单位净值
        - 日增长率: 日增长率(%)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 可购买: 是否可购买
        - 香港基金代码: 香港市场基金代码

    Examples:
        ```python
        await fund_hk_rank_em(
            condition="日增长率 > 0 AND 近1年 > 10"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_hk_rank_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_rating_all(condition: str = None) -> pd.DataFrame:
    """
    天天基金网-基金评级-基金评级总汇

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "5星评级家数 >= 3 AND 手续费 < 1.5",
            "基金公司 = '易方达'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 代码: 基金代码
        - 简称: 基金简称
        - 基金经理: 基金经理姓名
        - 基金公司: 基金公司名称
        - 5星评级家数: 获得5星评级的机构数量
        - 上海证券: 上海证券评级结果
        - 招商证券: 招商证券评级结果
        - 济安金信: 济安金信评级结果
        - 手续费: 申购费率(%)
        - 类型: 基金类型

    Examples:
        ```python
        await fund_rating_all(
            condition="5星评级家数 >= 3 AND 手续费 < 1.5"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_rating_all()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_rating_sh(date: str = "20230630", condition: str = None) -> pd.DataFrame:
    """
    天天基金网-基金评级-上海证券评级

    Input Schema:
    {
      "type": "object",
      "properties": {
        "date": {
          "type": "string",
          "description": "查询日期(yyyyMMdd格式)",
          "pattern": "^\\d{8}$",
          "examples": ["20230630", "20231231"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "3年期评级-3年评级 >= 4 AND 手续费 < 1.5",
            "近1年涨幅 > 10"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 代码: 基金代码
        - 简称: 基金简称
        - 基金经理: 基金经理姓名
        - 基金公司: 基金公司名称
        - 3年期评级-3年评级: 3年期评级结果(1-5星)
        - 3年期评级-较上期: 较上期评级变化(上升/下降/持平)
        - 5年期评级-5年评级: 5年期评级结果(1-5星)
        - 5年期评级-较上期: 较上期评级变化(上升/下降/持平)
        - 单位净值: 最新单位净值(元)
        - 日期: 净值日期
        - 日增长率: 单日增长率(%)
        - 近1年涨幅: 近1年收益率(%)
        - 近3年涨幅: 近3年收益率(%)
        - 近5年涨幅: 近5年收益率(%)
        - 手续费: 申购费率(%)
        - 类型: 基金类型

    Examples:
        ```python
        await fund_rating_sh(
            date="20230630",
            condition="3年期评级-3年评级 >= 4 AND 手续费 < 1.5"
        )
        ```

    Error Handling:
        - 如果日期格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_rating_sh(date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_rating_zs(date: str = "20230331", condition: str = None) -> pd.DataFrame:
    """
    天天基金网-基金评级-招商证券评级

    Input Schema:
    {
      "type": "object",
      "properties": {
        "date": {
          "type": "string",
          "description": "查询日期(yyyyMMdd格式)",
          "pattern": "^\\d{8}$",
          "examples": ["20230331", "20230630"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "3年期评级-3年评级 >= 4 AND 手续费 < 1.5",
            "近1年涨幅 > 10"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 代码: 基金代码
        - 简称: 基金简称
        - 基金经理: 基金经理姓名
        - 基金公司: 基金公司名称
        - 3年期评级-3年评级: 3年期评级结果(1-5星)
        - 3年期评级-较上期: 较上期评级变化(上升/下降/持平)
        - 单位净值: 最新单位净值(元)
        - 日期: 净值日期
        - 日增长率: 单日增长率(%)
        - 近1年涨幅: 近1年收益率(%)
        - 近3年涨幅: 近3年收益率(%)
        - 近5年涨幅: 近5年收益率(%)
        - 手续费: 申购费率(%)

    Examples:
        ```python
        await fund_rating_zs(
            date="20230331",
            condition="3年期评级-3年评级 >= 4 AND 手续费 < 1.5"
        )
        ```

    Error Handling:
        - 如果日期格式无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_rating_zs(date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_rating_ja(date: str = "20230331", condition: str = None) -> pd.DataFrame:
    """
    天天基金网-基金评级-济安金信评级
    Input Schema:
      {
        "type": "object",
        "properties": {
          "date": {
            "type": "string",
            "description": "查询日期(yyyyMMdd格式)",
            "pattern": "^\\d{8}$",
            "examples": ["20230331", "20230630"]
          },
          "condition": {
            "type": "string",
            "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
            "examples": [
              "3年期评级-3年评级 >= 4 AND 手续费 < 1.5",
              "近1年涨幅 > 10"
            ]
          }
        },
        "required": []
      }

    Returns:返回包含以下字段的DataFrame的Markdown格式:
            - 代码
            - 简称
            - 基金经理
            - 基金公司
            - 3年期评级-3年评级
            - 3年期评级-较上期
            - 单位净值
            - 日期
            - 日增长率
            - 近1年涨幅
            - 近3年涨幅
            - 近5年涨幅
            - 手续费
            - 类型

    """
    df = ak.fund_rating_ja(date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def stock_news_em(symbol: str = "300059") -> pd.DataFrame:
    """
    东方财富-个股新闻-最近 100 条新闻
    Args:
        symbol: 股票代码

    Returns:东方财富-个股新闻-最近 100 条新闻

    """
    return ak.stock_news_em(symbol).to_markdown()


@mcp.tool()
async def news_cctv(date: str = "20240424") -> pd.DataFrame:
    """
    新闻联播文字稿
    Args:
        date: 需要获取数据的日期; 目前 20160203 年后

    Returns:新闻联播文字稿

    """
    return ak.news_cctv(date).to_markdown()


def apply_filters_for_data_frame(
    df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    """
    通用Pandas筛选函数(字符串条件版)
    支持类似SQL WHERE子句的筛选条件语法

    示例:
        "pe > 20"
        "roe >= 0.15 AND turnoverRate < 5"
        "date >= '2023-01-01' AND close > open"
        "code in ['600000', '601318']"

    支持操作符:
        >, <, >=, <=, =, !=, in, not in, contains, not contains
    支持逻辑运算符:
        AND, OR, NOT
    支持括号分组:
        "(pe > 20 OR pe < 10) AND roe > 0.15"
    Args:
        df: 要筛选的DataFrame
        condition: 筛选条件字符串
    Returns:
        筛选后的DataFrame
    Raises:
        ValueError: 如果条件语法错误或列名不存在
    """
    if not condition:
        return df.copy()

    # 复制DataFrame避免修改原始数据
    df = df.copy()

    # 自动检测并转换数值列
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # 尝试转换为数值类型
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass

    # 解析条件字符串为Pandas查询表达式
    try:
        query_expr = _parse_condition_to_query(condition, df.columns)
        return df.query(query_expr)
    except Exception as e:
        raise ValueError(f"条件查询失败: {str(e)}") from e


def _parse_condition_to_query(condition: str, valid_columns) -> str:
    """
    将条件字符串解析为Pandas query表达式

    Args:
        condition: 原始条件字符串
        valid_columns: DataFrame的有效列名集合(可以是set或pandas Index)
    Returns:
        可用于DataFrame.query()的表达式字符串
    """
    import re
    from ast import literal_eval

    original_condition = condition

    # 确保valid_columns是set类型
    if not isinstance(valid_columns, set):
        valid_columns = set(valid_columns)

    # 预处理: 移除多余空格，统一操作符格式
    condition = re.sub(r"\s+", " ", condition.strip())

    # 定义逻辑运算符关键字
    LOGIC_KEYWORDS = {"AND", "OR", "NOT", "and", "or", "not"}

    # 改进的列名匹配模式，支持中文、特殊字符和百分号
    column_pattern = r"([a-zA-Z_\u4e00-\u9fa5][\w\u4e00-\u9fa5\-%]*)"
    columns = set(re.findall(column_pattern, condition)) - LOGIC_KEYWORDS

    # 检查列名是否有效
    invalid_cols = [col for col in columns if col not in valid_columns]
    if invalid_cols:
        raise ValueError(f"无效列名: {invalid_cols}")

    # 处理带引号的列名
    quoted_columns = re.findall(r'[\'"](.*?)[\'"]', condition)
    for col in quoted_columns:
        if col in valid_columns:
            condition = condition.replace(f"'{col}'", f"`{col}`").replace(
                f'"{col}"', f"`{col}`"
            )

    # 确保所有列名都被反引号包裹
    for col in columns:
        if col in valid_columns:
            # 处理包含特殊字符的列名
            if any(c in col for c in ["-", "%", " "]):
                # 获取原始列名(不带反引号)
                raw_col = col.strip("`")
                # 替换条件中的列名(带或不带反引号)
                condition = re.sub(
                    rf"(?<!\w)(`?{re.escape(raw_col)}`?)(?!\w)",
                    f"`{raw_col}`",
                    condition,
                )
            else:
                # 普通列名处理
                if (
                    f"`{col}`" not in condition
                    and f"'{col}'" not in condition
                    and f'"{col}"' not in condition
                ):
                    condition = re.sub(rf"(?<!\w){col}(?!\w)", f"`{col}`", condition)

    # 转换操作符 - 使用正则表达式确保精确匹配
    operator_patterns = [
        (r"(?<!\S)!=(?!\S)", "!="),  # != 保持不变
        (r"(?<!\S)=(?!\S)", "=="),  # = 替换为 ==
        (r"(?<!\S)>(?!\S)", ">"),  # > 保持不变
        (r"(?<!\S)<(?!\S)", "<"),  # < 保持不变
        (r"(?<!\S)>=(?!\S)", ">="),  # >= 保持不变
        (r"(?<!\S)<=(?!\S)", "<="),  # <= 保持不变
        (r"(?<!\S)in(?!\S)", ".isin("),  # in 替换为 .isin(
        (r"(?<!\S)not in(?!\S)", ".notin("),  # not in 替换为 .notin(
        (r"(?<!\S)contains(?!\S)", ".str.contains("),  # contains 替换
        (r"(?<!\S)not contains(?!\S)", ".str.contains("),  # not contains 替换
    ]

    # 处理字符串字面量
    condition = re.sub(r"'(.*?)'", r'"\1"', condition)  # 单引号转双引号

    # 处理IN表达式
    in_pattern = (
        r"(`?[a-zA-Z_\u4e00-\u9fa5][\w\u4e00-\u9fa5-%]*`?)\s+(in|not in)\s+(\[.*?\])"
    )

    def replace_in(match):
        col = match.group(1)
        # 处理包含特殊字符的列名
        if any(c in col for c in ["-", "%", " "]):
            # 确保列名被反引号包裹
            if not col.startswith("`") or not col.endswith("`"):
                col = f"`{col.strip('`')}`"
        else:
            # 普通列名处理
            if not col.startswith("`"):
                col = f"`{col}`"
        op = match.group(2)
        values = match.group(3)
        try:
            # 安全解析列表
            values = literal_eval(values)
            if not isinstance(values, list):
                raise ValueError
            values_str = str(values)
        except:
            raise ValueError(f"无效的列表格式: {values}")

        if op == "in":
            return f"`{col}`.isin({values_str})"
        else:
            return f"~`{col}`.isin({values_str})"

    condition = re.sub(in_pattern, replace_in, condition)

    # 处理逻辑运算符 - 统一转换为小写
    condition = re.sub(r"\bAND\b", "&", condition)
    condition = re.sub(r"\bOR\b", "|", condition)
    condition = re.sub(r"\bNOT\b", "~", condition)

    # 处理其他操作符
    for pattern, replacement in operator_patterns:
        condition = re.sub(pattern, replacement, condition)

    # 处理字符串包含操作
    condition = condition.replace(".str.contains(", ".str.contains(")
    condition = condition.replace(" not contains ", " not contains ")

    # 验证最终表达式语法
    try:
        # 使用空DataFrame测试语法(确保columns是list类型)
        test_df = pd.DataFrame(columns=list(valid_columns))
        test_df.query(condition)
    except Exception as e:
        # 提供更友好的错误提示
        error_msg = f"无效的条件表达式: {str(e)}\n"
        error_msg += f"原条件: {original_condition}\n"
        error_msg += f"转换后: {condition}\n"
        error_msg += "提示: 请检查列名、操作符和值格式是否正确"
        raise ValueError(error_msg) from e

    return condition


# 分位数计算, 1260对应5年的交易日
def __calculate_percentile(df, item_col_name, window=1260):
    df = df.copy()
    # 将date转换为DatetimeIndex
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # 数据清洗
    df[item_col_name] = df[item_col_name].replace([np.inf, -np.inf], np.nan)
    df[item_col_name] = df[item_col_name].interpolate(method="time").ffill().bfill()

    # 安全计算函数
    def safe_percentile(x):
        valid_data = x[~np.isnan(x)]
        if len(valid_data) < 60:
            return np.nan
        current = valid_data[-1]
        history = valid_data[:-1]
        return (current > history).mean() * 100

    # 滚动计算
    item_percentile = item_col_name + "_percentile"
    df[item_percentile] = (
        df[item_col_name]
        .rolling(window=window, min_periods=60)
        .apply(safe_percentile, raw=True)
    )
    # 处理缺失值
    df[item_percentile] = (
        df[item_percentile].interpolate(method="linear").ffill().bfill()
    )
    # 重置索引，恢复date列
    df = df.reset_index()
    return df


def __filter_abnormal_3delta(df, column_name):
    lower = df[column_name].mean() - 3 * df[column_name].std()
    upper = df[column_name].mean() + 3 * df[column_name].std()
    df = df[(df[column_name] > lower) & (df[column_name] < upper)]
    return df

    # if __name__ == "__main__":
    #     df = asyncio.run(
    #         fund_money_fund_daily_em(
    #             condition="上一个交易日-7日年化% >= 2.3 AND 手续费 < 0.8"
    #         )
    #     )
    # print(df)
    #     df = asyncio.run(get_stock_pe("沪深300", "pe_percentile > 10 and pe < 30"))
    # print(df)


if __name__ == "__main__":
    mcp.run()
