import asyncio
import datetime
import operator
import time
from typing import Dict, List, Union
from fastmcp import FastMCP
import pandas as pd
import akshare as ak
import numpy as np
from pydantic import Field

mcp = FastMCP("stock MCP", dependencies=["pandas", "numpy", "akshare"])

START_DATE = '20200101'
END_DATE = '20250418'


@mcp.tool()
async def get_single_stock_info(
        stock_code: str,
        start_date: str,
        end_date: str,
        retry: int = 3,
):
    """
    根据股票代码获取股票的历史数据，包含PE、ROE等
    Args:
       stock_code: 股票代码
       start_date: 开始日期，格式yyyyMMdd,
       end_date: 结束日期，格式yyyyMMdd,
       retry: 请求失败后的重试次数
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

                # === 获取财务数据（新版接口）===
                # 方法1：新浪财经接口（获取市盈率）
                pe_df = ak.stock_index_pe_lg()

                # 方法2：财报数据接口（获取ROE）
                finance_df = ak.stock_financial_abstract(symbol=stock_code)
                finance_df = finance_df[
                    finance_df['指标'].str.contains('ROE') & finance_df['选项'].str.contains('盈利能力')].copy()
                finance_df = finance_df.drop(columns=['选项'])
                report_date_cols = [col for col in finance_df.columns if col.isdigit and len(col) == 8]
                finance_df = pd.melt(finance_df, var_name='报表日期', value_name='净资产收益率(ROE)',
                                     value_vars=report_date_cols)
                finance_df['date'] = pd.to_datetime(finance_df['报表日期'])
                finance_df['roe'] = finance_df['净资产收益率(ROE)'] / 100

                # === 数据合并 ===
                # 合并PE数据
                price_df['date'] = pd.to_datetime(price_df['date'])
                pe_df['日期'] = pd.to_datetime(pe_df['日期'])
                """
                市盈率指标详解：
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
                滚动市盈率: 使用最新12个月数据，能更快反映公司近期的盈利变化，尤其适合：1、盈利波动大的行业（如周期股、科技股）。2、财报发布后的短期估值更新。
                静态市盈率的局限性: 静态数据容易滞后，尤其当公司近期盈利大幅增长或下滑时（如新能源、消费电子行业）。
                个股分析：优先用滚动市盈率（TTM），结合行业分位数（如当前PE处于历史30%分位）。
                市场估值判断：用等权滚动市盈率，避免被权重股扭曲。
                """
                merged_df = pd.merge_asof(
                    price_df.sort_values('date'),
                    pe_df[['日期', '等权滚动市盈率']].rename(columns={'日期': 'date', '等权滚动市盈率': 'pe'}),
                    on='date',
                    direction='backward'
                )

                # 合并ROE数据
                merged_df = pd.merge_asof(
                    merged_df.sort_values('date'),
                    finance_df[['date', 'roe']].sort_values('date'),
                    on='date',
                    direction='backward'
                )

                # === 数据清洗 ===
                merged_df['pe'] = merged_df['pe'].replace([np.inf, -np.inf], np.nan)
                merged_df['pe'] = merged_df['pe'].fillna(merged_df['pe'].median())
                merged_df['roe'] = merged_df['roe'].ffill()
                return merged_df
            except Exception as e:
                print(f"\n{stock_code}价格数据获取失败:{e}，重试中...")
                time.sleep(1)
    except Exception as e:
        print(f"{stock_code} 数据下载失败：{str(e)}\r\n")
        return None


@mcp.tool()
async def get_stock_indicator(stock_code: str,
                              indicator_type: str = "沪深300"
                              ):
    """
    根据股票代码获取指数历史行情数据，包含PE、PB和对应的百分位等
    Args:
        stock_code: 股票代码
        indicator_type: E、PB历史指标类型，根据股票所处的分类区分，可选值有：上证50, 沪深300, 上证380, 创业板50, 中证500, 上证180, 深证红利, 深证100, 中证1000, 上证红利, 中证100, 中证800

    Returns:
      指数型股票的历史行情数据，pandas的DataFrame形式
    """
    # 获取基础行情
    df_price = ak.stock_zh_index_daily(stock_code)
    df_price["date"] = pd.to_datetime(df_price["date"])

    # 估值数据获取
    # 市盈率数据
    df_pe = asyncio.run(get_stock_pe(indicator_type))
    # df_pe = get_stock_pe(indicator_type)
    df_pe["date"] = pd.to_datetime(df_pe["date"])

    # 市净率数据
    df_pb = asyncio.run(get_stock_pb(indicator_type))
    # df_pb = get_stock_pb(indicator_type)
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
    return merged_df


@mcp.tool()
async def get_stock_pe(stock_type: str = "沪深300"):
    """
    根据股票类型获取对应市场的历史PE数据(等权滚动市盈率)
    Args:
        stock_type: PE、PB历史指标类型，根据股票所处的分类区分，可选值有：上证50, 沪深300, 上证380, 创业板50, 中证500, 上证180, 深证红利, 深证100, 中证1000, 上证红利, 中证100, 中证800
    Returns: PE数据
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
async def get_stock_pb(stock_type: str = "沪深300"):
    """
        根据股票类型获取对应市场的历史PE数据
        Args:
            stock_type: PE、PB历史指标类型，根据股票所处的分类区分，可选值有：上证50, 沪深300, 上证380, 创业板50, 中证500, 上证180, 深证红利, 深证100, 中证1000, 上证红利, 中证100, 中证800
        Returns: PE数据
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
async def fund_purchase_em():
    """
    获取所有基金申购状态
    Returns: 基金申购状态
    """
    df = ak.fund_purchase_em()
    return df


@mcp.tool()
async def fund_name_em():
    """
    获取东方财富网站-天天基金网-基金数据-所有基金的名称和类型
    数据来源：https://fund.eastmoney.com/manager/default.html#dt14;mcreturnjson;ftall;pn20;pi1;scabbname;stasc
    Returns:
      所有基金的名称和类型
    """
    df = ak.fund_name_em()
    return df


@mcp.tool()
async def fund_info_index_em(
        symbol: str = "沪深指数", indicator: str = "被动指数型"
):
    """
    获取指数型基金近几年的单位净值、增长率、手续费等
    https://fund.eastmoney.com/trade/zs.html
    Args:
        symbol: 行业类型，可选值："全部", "沪深指数", "行业主题", "大盘指数", "中盘指数", "小盘指数", "股票指数", "债券指数"
        indicator: 指标类型，可选值："全部", "被动指数型", "增强指数型"
    Returns: 指数型基金近几年的单位净值、增长率、手续费等
    """
    df = ak.fund_info_index_em(symbol, indicator)
    return df


@mcp.tool()
async def fund_open_fund_info_em(
        symbol: str = "710001", indicator: str = "单位净值走势", period: str = "成立来"
):
    """
    获取指定基金特定指标的数据
    Args:
        symbol: 基金代码
        indicator: 需要获取的指标
        period: 期间，可选值：{"1月", "3月", "6月", "1年", "3年", "5年", "今年来", "成立来"}
    Returns:指定基金特定指标的数据
    """
    df = ak.fund_open_fund_info_em(symbol, indicator, period)
    return df


@mcp.tool()
async def fund_money_fund_daily_em():
    """
    获取当前交易日的所有货币型基金收益数据列表
    Returns: 当前交易日的所有货币型基金收益数据
    """
    df = ak.fund_money_fund_daily_em()
    return df


@mcp.tool()
async def fund_money_fund_info_em(symbol: str = "000009"):
    """
    获取指定的货币型基金收益-历史净值数据
    Args:
        symbol: 货币型基金代码, 可以通过 fund_money_fund_daily_em 来获取
    Returns: 货币型基金收益-历史净值数据
    """
    df = ak.fund_money_fund_info_em(symbol)
    return df


@mcp.tool()
async def fund_etf_fund_daily_em():
    """
    获取所有场内基金数据列表
    Returns: 当前交易日的所有场内交易基金数据
    """
    df = ak.fund_etf_fund_daily_em()
    return df


@mcp.tool()
async def fund_etf_fund_info_em(
        fund: str = "511280",
        start_date: str = "20000101",
        end_date: str = "20500101",
):
    """
    获取指定的场内交易基金的历史净值明细
    Args:
        fund: 场内交易基金代码, 可以通过 fund_etf_fund_daily_em 来获取
        start_date: 开始统计时间
        end_date: 结束统计时间
    Returns: 东方财富网站-天天基金网-基金数据-场内交易基金-历史净值明细
    """
    df = ak.fund_etf_fund_info_em(fund, start_date, end_date)
    return df


@mcp.tool()
async def fund_value_estimation_em(symbol: str = "全部"):
    """
    按照类型获取近期净值估算数据
    Args:
        symbol: 类型，可选值： {'全部', '股票型', '混合型', '债券型', '指数型', 'QDII', 'ETF联接', 'LOF', '场内交易基金'}
    Returns: 近期净值估算数据
    """
    df = ak.fund_value_estimation_em(symbol)
    return df


@mcp.tool()
async def fund_aum_em():
    """
    获取基金公司排名列表
    Returns:基金公司排名列表
    """
    return ak.fund_aum_em()


@mcp.tool()
async def fund_aum_trend_em() -> pd.DataFrame:
    """
    基金市场管理规模走势图
    Returns: 基金市场管理规模走势图
    """
    return ak.fund_aum_trend_em()


@mcp.tool()
async def fund_aum_hist_em(year: str = "2023") -> pd.DataFrame:
    """
    金公司历年管理规模排行列表
    Args:
        year: 年份，如：2024
    Returns: 金公司历年管理规模排行列表
    """
    return ak.fund_aum_hist_em(year)


@mcp.tool()
async def fund_announcement_personnel_em(symbol: str = "000001") -> pd.DataFrame:
    """
    基金的人事调整-公告列表
    Args:
        symbol: 基金代码; 可以通过调用 ak.fund_name_em() 接口获取
    Returns:  基金的人事调整-公告列表
    """
    return ak.fund_announcement_personnel_em(symbol)


@mcp.tool()
async def fund_etf_spot_em() -> pd.DataFrame:
    """
    ETF 实时行情
    Returns: ETF 实时行情
    """
    return ak.fund_etf_spot_em()


@mcp.tool()
async def fund_etf_hist_em(
        symbol: str = "159707",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        adjust: str = "",
) -> pd.DataFrame:
    """
    获取特定基金的每日ETF行情数据
    Args:
        symbol: ETF 代码
        period: 类型，选项有：{'daily', 'weekly', 'monthly'}
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式，选项有：{"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    Returns:特定基金的每日ETF行情数据
    """
    return ak.fund_etf_hist_em(symbol, period, start_date, end_date, adjust)


@mcp.tool()
async def fund_etf_hist_min_em(
        symbol: str = "159707",
        start_date: str = "1979-09-01 09:32:00",
        end_date: str = "2222-01-01 09:32:00",
        period: str = "5",
        adjust: str = "",
) -> pd.DataFrame:
    """
    获取特定基金的时分ETF行情数据
    Args:
        symbol: ETF 代码
        start_date: 开始日期
        end_date: 结束日期
        period: 时间间隔类型，选项：{"1", "5", "15", "30", "60"}
        adjust: 复权方式，选项：{'', 'qfq', 'hfq'}
    Returns: 特定基金的时分ETF行情数据
    """
    return ak.fund_etf_hist_min_em(symbol, period, start_date, end_date, adjust)


@mcp.tool()
async def fund_fee_em(symbol: str = "015641", indicator: str = "认购费率") -> pd.DataFrame:
    """
    基金的交易规则，费率等
    Args:
        symbol: 基金代码
        indicator: 指标，可选值：{"交易状态", "申购与赎回金额", "交易确认日", "运作费用", "认购费率", "申购费率", "赎回费率"}
    Returns: 基金的交易规则，费率等
    """
    return ak.fund_fee_em(symbol, indicator)


@mcp.tool()
async def fund_etf_category_sina(symbol: str = "LOF基金") -> pd.DataFrame:
    """
    指定类型的基金列表
    Args:
        symbol: 类型，可选值有："封闭式基金", "ETF基金", "LOF基金"

    Returns: 指定类型的基金列表

    """
    return ak.fund_etf_category_sina(symbol)


@mcp.tool()
async def fund_etf_hist_sina(symbol: str = "sh510050") -> pd.DataFrame:
    """
    基金的日行情数据
    Args:
        symbol: 基金名称, 可以通过 ak.fund_etf_category_sina() 函数获取
    Returns: 基金的日行情数据
    """
    return ak.fund_etf_hist_sina(symbol)


@mcp.tool()
async def fund_etf_dividend_sina(symbol: str = "sh510050") -> pd.DataFrame:
    """
    基金的累计分红数据
    Args:
        symbol: 基金名称, 可以通过 ak.fund_etf_category_sina() 函数获取
    Returns: 基金的累计分红数据
    """
    return ak.fund_etf_dividend_sina(symbol)


@mcp.tool()
async def fund_etf_spot_ths(date: str = "") -> pd.DataFrame:
    """
    同花顺理财-基金数据-每日净值-ETF-实时行情
    Args:
        date: 交易日期
    Returns: 每日净值-ETF-实时行情
    """
    return ak.fund_etf_spot_ths(date)


@mcp.tool()
async def fund_individual_basic_info_xq(
        symbol: str = "000001", timeout: float = None
) -> pd.DataFrame:
    """
    基金基本信息
    Args:
        symbol: 基金代码
        timeout: 超时时间

    Returns: 基金基本信息

    """
    return ak.fund_individual_basic_info_xq(symbol, timeout)


@mcp.tool()
async def fund_individual_achievement_xq(
        symbol: str = "000001", timeout: float = None
) -> pd.DataFrame:
    """
    基金业绩
    Args:
        symbol: 基金代码
        timeout: 超时时间

    Returns:基金业绩

    """
    return ak.fund_individual_achievement_xq(symbol, timeout)


@mcp.tool()
async def fund_individual_analysis_xq(
        symbol: str = "000001", timeout: float = None
) -> pd.DataFrame:
    """
    基金数据分析
    Args:
        symbol: 基金代码
        timeout: 超时时间

    Returns: 基金数据分析

    """
    return ak.fund_individual_analysis_xq(symbol, timeout)


@mcp.tool()
async def fund_individual_profit_probability_xq(
        symbol: str = "000001", timeout: float = None
) -> pd.DataFrame:
    """
    雪球基金-盈利概率-历史任意时点买入，持有满 X 年，盈利概率 Y%
    Args:
        symbol: 基金代码
        timeout: 超时时间

    Returns:盈利概率

    """
    return ak.fund_individual_profit_probability_xq(symbol, timeout)


@mcp.tool()
async def fund_individual_detail_info_xq(
        symbol: str = "000001", timeout: float = None
) -> pd.DataFrame:
    """
    雪球基金-交易规则
    Args:
        symbol: 基金代码
        timeout: 超时时间

    Returns:交易规则

    """
    return ak.fund_individual_detail_info_xq(symbol, timeout)


@mcp.tool()
async def fund_individual_detail_hold_xq(
        symbol: str = "002804", date: str = "20231231", timeout: float = None
) -> pd.DataFrame:
    """
    雪球基金-持仓
    Args:
        symbol: 基金代码
        date: 财报日期
        timeout: 超时时间

    Returns:

    """
    return ak.fund_individual_detail_hold_xq(symbol, date, timeout)


@mcp.tool()
async def fund_manager_em() -> pd.DataFrame:
    """
    天天基金网-基金数据-基金经理大全
    Returns:天天基金网-基金数据-基金经理大全

    """
    return ak.fund_manager_em()


@mcp.tool()
async def fund_portfolio_hold_em(symbol: str = "000001", date: str = "2024") -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-基金持仓
    Args:
        symbol: 基金代码
        date: 查询年份

    Returns: 天天基金网-基金档案-投资组合-基金持仓

    """
    return ak.fund_portfolio_hold_em(symbol, date)


@mcp.tool()
async def fund_portfolio_bond_hold_em(
        symbol: str = "000001", date: str = "2023"
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-债券持仓
    Args:
        symbol: 基金代码
        date: 查询年份

    Returns:天天基金网-基金档案-投资组合-债券持仓

    """
    return ak.fund_portfolio_bond_hold_em(symbol, date)


@mcp.tool()
async def fund_portfolio_industry_allocation_em(
        symbol: str = "000001", date: str = "2023"
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-行业配置
    Args:
        symbol: 基金代码
        date: 查询年份

    Returns:天天基金网-基金档案-投资组合-行业配置

    """
    return ak.fund_portfolio_industry_allocation_em(symbol, date)


@mcp.tool()
async def fund_portfolio_change_em(
        symbol: str = "003567", indicator: str = "累计买入", date: str = "2023"
) -> pd.DataFrame:
    """
    天天基金网-基金档案-投资组合-重大变动
    Args:
        symbol: 基金代码
        indicator: 指标，可选值："累计买入", "累计卖出"
        date: 查询年份

    Returns:

    """
    return ak.fund_portfolio_change_em(symbol, indicator, date)


@mcp.tool()
async def fund_stock_position_lg() -> pd.DataFrame:
    """
    乐咕乐股-基金仓位-股票型基金仓位
    Returns:乐咕乐股-基金仓位-股票型基金仓位

    """
    return ak.fund_stock_position_lg()


@mcp.tool()
async def fund_balance_position_lg() -> pd.DataFrame:
    """
    乐咕乐股-基金仓位-平衡混合型基金仓位
    Returns: 乐咕乐股-基金仓位-平衡混合型基金仓位

    """
    return ak.fund_balance_position_lg()


@mcp.tool()
async def fund_linghuo_position_lg() -> pd.DataFrame:
    """
    乐咕乐股-基金仓位-灵活配置型基金仓位
    Returns:乐咕乐股-基金仓位-灵活配置型基金仓位

    """
    return ak.fund_linghuo_position_lg()


@mcp.tool()
async def fund_open_fund_rank_em(symbol: str = "全部") -> pd.DataFrame:
    """
    东方财富网-数据中心-开放基金排行
    Args:
        symbol: 基金类型，可选值："全部", "股票型", "混合型", "债券型", "指数型", "QDII", "FOF"

    Returns: 开放基金排行

    """
    return ak.fund_open_fund_rank_em(symbol)


@mcp.tool()
async def fund_exchange_rank_em() -> pd.DataFrame:
    """
    东方财富网-数据中心-场内交易基金排行
    Returns: 东方财富网-数据中心-场内交易基金排行

    """
    return ak.fund_exchange_rank_em()


@mcp.tool()
async def fund_money_rank_em() -> pd.DataFrame:
    """
    东方财富网-数据中心-货币型基金排行
    Returns:
东方财富网-数据中心-货币型基金排行
    """
    return ak.fund_money_rank_em()


@mcp.tool()
async def fund_lcx_rank_em() -> pd.DataFrame:
    """
    东方财富网-数据中心-理财基金排行
    Returns:东方财富网-数据中心-理财基金排行

    """
    return ak.fund_lcx_rank_em()


@mcp.tool()
async def fund_hk_rank_em() -> pd.DataFrame:
    """
    东方财富网-数据中心-香港基金排行
    Returns:东方财富网-数据中心-香港基金排行

    """
    return ak.fund_hk_rank_em()


@mcp.tool()
async def fund_rating_all() -> pd.DataFrame:
    """
    天天基金网-基金评级-基金评级总汇
    Returns: 天天基金网-基金评级-基金评级总汇

    """
    return ak.fund_rating_all()


@mcp.tool()
async def fund_rating_sh(date: str = "20230630") -> pd.DataFrame:
    """
    天天基金网-基金评级-上海证券评级
    Args:
        date: 日期; https://fund.eastmoney.com/data/fundrating_3.html 获取查询日期

    Returns:天天基金网-基金评级-上海证券评级

    """
    return ak.fund_rating_sh(date)


@mcp.tool()
async def fund_rating_zs(date: str = "20230331") -> pd.DataFrame:
    """
    天天基金网-基金评级-招商证券评级
    Args:
        date: 日期; https://fund.eastmoney.com/data/fundrating_2.html 获取查询日期

    Returns:天天基金网-基金评级-招商证券评级

    """
    return ak.fund_rating_zs(date)


@mcp.tool()
async def fund_rating_ja(date: str = "20230331") -> pd.DataFrame:
    """
    天天基金网-基金评级-济安金信评级
    Args:
        date: 日期; https://fund.eastmoney.com/data/fundrating_4.html 获取查询日期

    Returns:天天基金网-基金评级-济安金信评级

    """
    return ak.fund_rating_ja(date)


@mcp.tool()
async def stock_news_em(symbol: str = "300059") -> pd.DataFrame:
    """
    东方财富-个股新闻-最近 100 条新闻
    Args:
        symbol: 股票代码

    Returns:东方财富-个股新闻-最近 100 条新闻

    """
    return ak.stock_news_em(symbol)


@mcp.tool()
async def news_cctv(date: str = "20240424") -> pd.DataFrame:
    """
    新闻联播文字稿
    Args:
        date: 需要获取数据的日期; 目前 20160203 年后

    Returns:新闻联播文字稿

    """
    return ak.news_cctv(date)

@mcp.tool()
async def apply_filters_for_data_frame(
        df: Union[dict, list],  # 接受字典或列表作为输入
        filters: List[Dict[str, Union[str, int, float, List, bool]]],
        operator_type: str = "and",
) -> dict:  # 返回字典而不是DataFrame
    """
    通用Pandas筛选函数
    筛选条件字典格式:
            {
                "column": "列名",
                "operator": "操作符",  # 可选值: eq, ne, gt, lt, ge, le, in, not_in, contains, not_contains
                "value": 比较值,
                "data_type": "数据类型"  # 可选: 'str', 'int', 'float', 'date'（可选，会自动推断）
            }
        使用示例1：简单的AND条件
            filters = [
                {"column": "Department", "operator": "eq", "value": "HR"},
                {"column": "Age", "operator": "gt", "value": 30}
            ]
            filtered_df = apply_filters_for_data_frame(df, filters)
        使用示例2：OR条件
            filters = [
                {"column": "Name", "operator": "contains", "value": "a"},
                {"column": "Salary", "operator": "ge", "value": 70000}
            ]
            filtered_df = apply_filters_for_data_frame(df, filters, operator_type="or")
         使用示例3：IN操作
            filters = [
                {"column": "Name", "operator": "in", "value": ["Alice", "Bob", "Charlie"]},
                {"column": "Join_Date", "operator": "gt", "value": "2020-01-01", "data_type": "date"}
            ]
            filtered_df = apply_filters_for_data_frame(df, filters)
    Args:
        df: 要筛选的DataFrame
        filters: 筛选条件列表，每个条件是一个字典
        operator_type: 多个条件之间的逻辑关系，"and"或"or"
    Returns: 筛选之后的DataFrame
    """
    # 将输入转换为DataFrame
    df = pd.DataFrame(df)
    
    if not filters:
        return df.to_dict(orient='records')

    # 定义操作符映射
    operator_map = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "lt": operator.lt,
        "ge": operator.ge,
        "le": operator.le,
        "in": lambda x, y: x.isin(y),
        "not_in": lambda x, y: ~x.isin(y),
        "contains": lambda x, y: x.str.contains(y),
        "not_contains": lambda x, y: ~x.str.contains(y),
    }
    mask_list = []
    for filter_ in filters:
        col = filter_["column"]
        op = filter_.get("operator", "eq")
        value = filter_["value"]
        data_type = filter_.get("data_type")
        # 自动推断数据类型
        if data_type is None:
            if isinstance(value, (list, tuple)):
                data_type = type(value[0]).__name__ if value else "str"
            else:
                data_type = type(value).__name__
        # 处理日期类型（需要转换为datetime）
        if data_type == "date" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
            if not isinstance(value, (list, tuple)):
                value = pd.to_datetime(value)
        # 获取操作符函数
        op_func = operator_map.get(op)
        if op_func is None:
            raise ValueError(f"不支持的运算符: {op}")
        # 应用筛选条件
        mask = op_func(df[col], value)
        mask_list.append(mask)
    # 组合多个条件
    if operator_type == "and":
        final_mask = pd.concat(mask_list, axis=1).all(axis=1)
    elif operator_type == "or":
        final_mask = pd.concat(mask_list, axis=1).any(axis=1)
    else:
        raise ValueError("operator_type 必须是 'and' 或 'or'")
    result_df = df[final_mask].copy()
    return result_df.to_dict(orient='records')


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
#     df = asyncio.run(get_stock_indicator("sh600006"))
#     print(df.head(10))

if __name__ == "__main__":
    mcp.run()
