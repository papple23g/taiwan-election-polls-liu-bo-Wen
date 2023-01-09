import datetime
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from typing_extensions import Literal


class ElectionPolls:
    """ 選舉民調
    """

    # 選舉民調網頁
    url: str = None
    # 民調資料表 table 在選舉民調網頁中的索引
    table_index: int = None
    # 選舉結果
    result_support_rate_dict: Dict[str, float]
    # 選舉日期
    result_date: datetime.date = None

    @classmethod
    def get_person_name_list(cls) -> List[str]:
        """ 獲取候選人名單

        Returns:
            List[str]: 候選人名單
        """
        return list(cls.result_support_rate_dict.keys())

    @classmethod
    def get_result_support_percent_arr(cls) -> np.ndarray:
        """ 獲取候選人支持率佔比

        Returns:
            np.ndarray
        """
        result_support_rate_arr = np.array(
            list(cls.result_support_rate_dict.values())
        )
        return result_support_rate_arr / sum(result_support_rate_arr)

    @classmethod
    def get_html_str(cls) -> str:
        """ 獲取網頁內容

        Returns:
            str: 網頁內容
        """
        html_path = Path(__file__).parent/f'html/{cls.__name__}.html'

        if html_path.exists():
            with html_path.resolve().open('r', encoding='utf8') as f:
                return f.read()

        with html_path.resolve().open('w', encoding='utf8') as f:
            response = requests.get(cls.url)
            f.write(response.text)
        return response.text

    @classmethod
    def get_raw_df(cls) -> pd.DataFrame:
        """ 獲取網頁上的民調原始資料表格資料

        Returns:
            pd.DataFrame
        """

        def is_not_content_row(sr: np.ndarray) -> bool:
            """ 判斷是否為非內容列

            Args:
                sr (np.array): 欄位值陣列

            Returns:
                bool: 是否為非內容列
            """
            if "%" not in sr[-1].replace("％", "%"):
                return True
            return all([
                v == sr[0]
                for v in sr[1:]
            ])

        # 獲取原始資料表
        html_str = cls.get_html_str()
        raw_df_list: List[pd.DataFrame] = pd.read_html(html_str)

        # 根據 table_index 或 docstring 來獲取民調資料表
        if cls.table_index is None:
            raw_df = next(
                (
                    raw_df
                    for raw_df in raw_df_list
                    if cls.__doc__.strip() in raw_df.columns
                ), None
            )
            if raw_df is None:
                raise ValueError("未找到民調資料表")
        else:
        raw_df = raw_df_list[cls.table_index]

        # 重整欄位列表
        raw_df.columns = [cols[-1] for cols in raw_df.columns]

        # 移除空資料的欄位
        raw_df = raw_df.dropna(axis=1)

        # 僅擷取全國民調
        raw_df: pd.DataFrame = raw_df.iloc[
            :raw_df[
                raw_df.apply(
                    is_not_content_row,
                    axis=1,
                )
            ].first_valid_index(),
        ]

        return raw_df

    @classmethod
    def format_survey_date_str(cls, survey_date_str: str) -> str:
        """ 格式化民調日期字串，如: '2020年01月01日' -> '2020-01-01'

        Args:
            survey_date_str (str): 民調日期字串

        Returns:
            str
        """
        return (
            survey_date_str
            .replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
        )

    @classmethod
    def get_df(
        cls,
        survey_support_rate_in_percent_bool: bool = False
    ) -> pd.DataFrame:
        """ 獲取處理後的民調資料

        Args:
            survey_support_rate_in_percent_bool (bool, optional):
                是否將民調支持率轉換為百分比文字. Defaults to False.

        Returns:
            pd.DataFrame: 包含以下欄位:
                調查單位: 調查組織名稱
                民調準確度: 民調與選舉結果的餘弦相似度
                [候選人名稱...]
        """

        def get_survey_percent_arr_and_cos_similarity(
            row_dict: Dict[str, str]
        ) -> Tuple[float, float, float, float]:
            """ 獲取歸一化後的民調百分比以及和選舉結果的 [餘弦相似度]

            Returns:
                Tuple[float, float, float, float]
            """
            survey_support_rate_arr = np.array([
                float(
                    row_dict[key]
                    .replace("％", "")
                    .replace("%", "")
                )
                for key in row_dict.keys()
                if any(
                    kw in key for kw in cls.result_support_rate_dict.keys()
                )
            ])
            survey_support_percent_arr = survey_support_rate_arr / \
                sum(survey_support_rate_arr)

            result_support_percent_arr = cls.get_result_support_percent_arr()

            cos_similarity = np.dot(
                survey_support_rate_arr, result_support_percent_arr
            ) / (
                np.linalg.norm(survey_support_rate_arr) *
                np.linalg.norm(result_support_percent_arr)
            )

            return (
                *survey_support_percent_arr.tolist(),
                cos_similarity,
            )

        # 獲取委託調查單位: 去除空白、後綴數字、圓括號附註內容
        raw_df = cls.get_raw_df()
        df = pd.DataFrame()
        df["調查單位"] = raw_df[raw_df.columns[0]].map(
            lambda unit_str: (
                re.sub(
                    r"\d+$",
                    "",
                    (
                        cast(str, unit_str)
                        .split("（")[0] if "（" in unit_str else unit_str
                    )
                    .replace(" ", "")
                    # 處理同單位變名
                    .replace("東森新聞", "東森")
                    .replace("東森", "東森/ETtoday")
                    .replace("ETtoday新聞雲", "東森/ETtoday")

                    .replace("中國時報", "中時")
                    .replace("旺旺中時", "中時")

                    .replace("蘋果新聞網", "蘋果")
                    .replace("蘋果日報", "蘋果")

                    .replace("民調", "")
                    .replace("臺灣", "台灣")
                    .replace("決策／精湛", "決策")
                    .replace("世新大學[1]", "世新大學")
                    .replace("民主進步黨", "民進黨")
                    .replace("中國國民黨", "國民黨")
                    .replace("台湾指标", "台灣指標")
                    .replace("聯合報系", "聯合報")
                    .replace("美麗島電子報", "美麗島")
                )
            )
        )
        print(df["調查單位"])

        # 獲取調查結束時間
        df["調查時間"] = pd.to_datetime(
            raw_df[raw_df.columns[1]].map(cls.format_survey_date_str)
        )

        # 獲取調查結束時間離選舉的天數
        df["選舉倒數"] = df["調查時間"].map(
            lambda dt: (
                cls.result_date
                - cast(datetime.datetime, dt).date()
            ).days
        )

        # 簡化候選人欄位名稱
        for person_name in cls.result_support_rate_dict.keys():
            for raw_col_name in raw_df.columns:
                if person_name in raw_col_name:
                    df[person_name] = raw_df[raw_col_name]

        # 計算各候選人佔票比例 以及 [餘弦相似度]
        *survey_percent_arr_list, df["民調準確度"] = zip(
            *df.apply(
                get_survey_percent_arr_and_cos_similarity,
                axis=1,
            )
        )
        for person_name, survey_percent_arr in zip(
            cls.result_support_rate_dict.keys(), survey_percent_arr_list
        ):
            df[person_name] = survey_percent_arr
            df[person_name] = df[person_name].map(
                lambda percent: (
                    f"{percent:.2%}" if survey_support_rate_in_percent_bool else
                    round(percent, 3)
                )
            )
        df["民調準確度"] = df["民調準確度"].map(
            lambda cos_similarity: round(cos_similarity, 5)
        )

        # 排除重複的調查單位，只留下該單位的最新一筆資料 (封關民調)
        df.sort_values(by="調查時間", ascending=False, inplace=True)
        df.drop_duplicates(
            subset=["調查單位"],
            inplace=True,
        )

        # 根據 [餘弦相似度] 排序資料
        df.sort_values(by="民調準確度", ascending=False, inplace=True)

        # 給予排名欄位
        df.insert(
            loc=0,
            column='排名',
            value=range(1, len(df) + 1),
        )

        df["調查時間"] = df["調查時間"].map(lambda date: date.date())

        return df

    @classmethod
    def plot_ternary(cls):
        """ 繪製三元相圖點陣圖
        """

        df = cls.get_df()

        fig = px.scatter_ternary(
            df,
            a=cls.get_person_name_list()[0],
            b=cls.get_person_name_list()[1],
            c=cls.get_person_name_list()[2],
            text="調查單位",
            size=df["選舉倒數"],
        )

        fig.add_trace(
            go.Scatterternary(
                a=[cls.get_result_support_percent_arr()[0]],
                b=[cls.get_result_support_percent_arr()[1]],
                c=[cls.get_result_support_percent_arr()[2]],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    size=20,
                ),
                text=["選舉結果"],
                textfont=dict(
                    color="red",
                )
            )
        )

        fig.update_traces(textposition='bottom center')

        # 調整繪圖範圍
        margin_percent = 0.01
        fig.update_ternaries(
            aaxis=dict(
                min=min([
                    df[cls.get_person_name_list()[0]].min(),
                    cls.get_result_support_percent_arr()[0],
                ])-margin_percent,
            ),
            baxis=dict(
                min=min([
                    df[cls.get_person_name_list()[1]].min(),
                    cls.get_result_support_percent_arr()[1],
                ])-margin_percent,
            ),
            caxis=dict(
                min=min([
                    df[cls.get_person_name_list()[2]].min(),
                    cls.get_result_support_percent_arr()[2],
                ])-margin_percent,
            ),
        )

        # 匯出 html 檔案並開啟
        html_path = (
            Path(__file__).parent / 'chart_html'
            / f"{cls.__name__}_ternary.html"
        )
        fig.write_html(html_path)
        os.startfile(html_path)

    @classmethod
    def plot_scatter(cls):
        """ 繪製二元點陣圖
        """

        df = cls.get_df()

        get_person_name_list = cls.get_person_name_list()
        assert len(get_person_name_list) == 2, "只能繪製兩個候選人的民調"
        person_name_a, person_name_b = cls.get_person_name_list()

        df['person_a_support_rate'] = df[person_name_a] / (
            df[person_name_b]+df[person_name_a]
        )
        result_person_a_support_rate = (
            cls.result_support_rate_dict[person_name_a] / (
                cls.result_support_rate_dict[person_name_b] +
                cls.result_support_rate_dict[person_name_a]
            )
        )
        fig = px.scatter(
            df,
            x="person_a_support_rate",
            y="民調準確度",
            text="調查單位",
            size=df["選舉倒數"],
            size_max=60,
        )

        fig.add_trace(
            go.Scatter(
                x=[result_person_a_support_rate, result_person_a_support_rate],
                y=[df['民調準確度'].min(), 1.005],
                mode='lines',
                line=dict(
                    color='rgb(255, 0, 0)',
                    width=2,
                    dash="dash",
                ),
            )
        )

        fig.add_annotation(
            text="選舉結果",
            x=result_person_a_support_rate,
            y=1.006,
            showarrow=False,
            font=dict(
                color='red',
                size=15,
            ),
        )

        fig.update_traces(textposition='top center')

        fig.update_layout(
            title=(
                f'{cls.__doc__.strip()} '
                f'({person_name_b} vs {person_name_a})'
            ),
            xaxis_title=f'{person_name_a} 佔票比例',
            yaxis_title='民調準確度',
        )

        # 匯出 html 檔案並開啟
        html_path = (
            Path(__file__).parent / 'chart_html'
            / f"{cls.__name__}_scatter.html"
        )
        fig.write_html(html_path)
        os.startfile(html_path)

    @classmethod
    def plot_survey_table(cls):
        """ 繪製民調結果表格
        """

        df = cls.get_df()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.03,
            specs=[
                [{"type": "table"}],
                [{"type": "table"}],
            ]
        )

        fig.add_trace(
            go.Table(
                header=dict(values=list(
                    cls.result_support_rate_dict.keys())),
                cells=dict(
                    values=[
                        round(result_support_percent, 3)
                        for result_support_percent in cls.get_result_support_percent_arr()
                    ]
                ),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Table(
                header=dict(values=df.columns.tolist()),
                cells=dict(values=df.values.T.tolist()),
            ),
            row=2, col=1
        )

        # 匯出 html 檔案並開啟
        html_path = (
            Path(__file__).parent / 'chart_html'
            / f"{cls.__name__}_survey_table.html"
        )
        fig.write_html(html_path)
        os.startfile(html_path)


class ElectionPollsPresident2012(ElectionPolls):
    """ 2012年中華民國總統選舉全國民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2012年中華民國總統選舉全國民意調查#候選人支持度"
    table_index = 0
    result_date = datetime.date(2012, 1, 14)
    result_support_rate_dict = {
        "蔡英文": 45.63,
        "馬英九": 51.60,
        "宋楚瑜": 2.77,
    }


class ElectionPollsPresident2016(ElectionPolls):
    """ 2016年中華民國總統選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2016年中華民國總統選舉民意調查#洪秀柱廢止提名後"
    table_index = 2
    result_date = datetime.date(2016, 1, 16)
    result_support_rate_dict = {
        "蔡英文": 56.12,
        "朱立倫": 31.04,
        "宋楚瑜": 12.83,
    }


class ElectionPollsTaipei2018(ElectionPolls):
    """ 2018年臺北市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_臺北市"
    table_index = 2
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "姚文智": 17.28,
        "丁守中": 40.81,
        "柯文哲": 41.06,
    }


class ElectionPollsNewTaipei2018(ElectionPolls):
    """ 2018年新北市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_新北市"
    table_index = 5
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "蘇貞昌": 42.85,
        "侯友宜": 57.14,
    }


class ElectionPollsTaoyuan2018(ElectionPolls):
    """ 2018年桃園市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_桃園市"
    table_index = 8
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "鄭文燦": 53.46,
        "陳學聖": 39.41,
    }


class ElectionPollsTaiChung2018(ElectionPolls):
    """ 2018年台中市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_台中市"
    table_index = 10
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "林佳龍": 43.34,
        "盧秀燕": 56.56,
    }


class ElectionPollsTainan2018(ElectionPolls):
    """ 2018年臺南市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_臺南市"
    table_index = 13
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "黃偉哲": 38.01,
        "高思博": 32.36,
    }


class ElectionPollsKaohsiung2018(ElectionPolls):
    """ 2018年高雄市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_高雄市"
    table_index = 16
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "陳其邁": 44.79,
        "韓國瑜": 53.86,
    }


class ElectionPollsPresident2020(ElectionPolls):
    """ 2020年中華民國總統選舉民意調查 
    """
    url = "https://zh.wikipedia.org/wiki/2020年中華民國總統選舉民意調查#英德配－國政配－瑜湘配"
    table_index = 0
    result_date = datetime.date(2020, 1, 11)
    result_support_rate_dict = {
        "蔡英文": 57.13,
        "韓國瑜": 38.61,
        "宋楚瑜": 4.26,
    }

    @classmethod
    def format_survey_date_str(cls, survey_date_str: str) -> str:
        _, survey_date_str = survey_date_str.split("－")
        survey_date_m, survey_date_d = survey_date_str.split("-")
        return f"2019-{survey_date_m}-{survey_date_d}"


def plot_survey_ranking_table(
    by: Literal['mean', 'median'] = 'median',
):
    """ 繪製民調結果準確度排名表格

    Args:
        by: 排名方式, 'mean' 或 'median'
    """
    survey_score_df = pd.DataFrame()
    for ElectionPoll in ElectionPolls.__subclasses__():
        sub_df = ElectionPoll.get_df()
        survey_score_df = pd.concat(
            [
                survey_score_df,
                sub_df[
                    [
                        "調查單位",
                        "民調準確度",
                    ]
                ],
            ]
        )

    survey_score_mean_df = survey_score_df.groupby("調查單位").agg(
        **{
            '民調準確度(中位數)': ('民調準確度', 'median'),
            '民調準確度(平均值)': ('民調準確度', 'mean'),
            '樣本數': ('民調準確度', 'count'),
        },
    )

    survey_score_mean_df.reset_index(inplace=True)

    survey_score_mean_df.sort_values(
        by=(
            "民調準確度(中位數)" if (by == 'median') else
            "民調準確度(平均值)"
        ),
        ascending=False,
        inplace=True,
    )

    # 將分數表為小數點後7位數
    survey_score_mean_df["民調準確度(中位數)"] = survey_score_mean_df["民調準確度(中位數)"].map(
        lambda x: f"{x:.7f}",
    )
    survey_score_mean_df["民調準確度(平均值)"] = survey_score_mean_df["民調準確度(平均值)"].map(
        lambda x: f"{x:.7f}",
    )

    # 給予排名欄位
    survey_score_mean_df.insert(
        loc=0,
        column='排名',
        value=range(1, len(survey_score_mean_df) + 1),
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=survey_score_mean_df.columns,
                ),
                cells=dict(
                    values=survey_score_mean_df.T.values,
                )
            ),
        ],
    )

    # 匯出 html 檔案並開啟
    html_path = (
        Path(__file__).parent / 'chart_html'
        / f"survey_ranking_table.html"
    )
    fig.write_html(html_path)
    os.startfile(html_path)
