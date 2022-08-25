import datetime
import random
import re
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import requests


class ElectionPolls:
    """ 選舉民調
    """

    # 選舉民調網頁
    url: str = None
    # 民調資料表 table 在選舉民調網頁中的索引
    table_index: int = 0
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
    def get_result_support_rate_uarr(cls) -> np.ndarray:
        """ 獲取候選人支持率陣列

        Returns:
            np.ndarray: 候選人支持率陣列
        """
        result_support_rate_arr = np.array(
            list(cls.result_support_rate_dict.values())
        )
        return result_support_rate_arr / (
            np.linalg.norm(result_support_rate_arr)
        )

    @classmethod
    def get_html_str(cls) -> str:
        """ 獲取網頁內容

        Returns:
            str: 網頁內容
        """
        html_path = Path(__file__)/f'../html/{cls.__name__}.html'

        if html_path.exists():
            with html_path.open('r', encoding='utf8') as f:
                return f.read()

        with html_path.open('w', encoding='utf8') as f:
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
            """ 判斷是否為內容列

            Args:
                sr (np.array): 欄位值陣列

            Returns:
                bool: 是否為內容列
            """
            if "%" not in sr[-1].replace("％", "%"):
                return True
            return all([
                v == sr[0]
                for v in sr[1:]
            ])

        html_str = cls.get_html_str()
        raw_df_list: List[pd.DataFrame] = pd.read_html(html_str)
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
    def formate_survey_date_str(cls, survey_date_str: str) -> str:
        return (
            survey_date_str
            .replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
        )

    @classmethod
    def get_df(cls) -> pd.DataFrame:
        """ 獲取處理後的民調資料

        Returns:
            pd.DataFrame: 包含以下欄位:
                ORG: 調查組織名稱
                cos_similarity: 民調與選舉結果的餘弦相似度
                [候選人名稱...]
        """

        def get_survey_rate_uarr_and_cos_similarity(row_dict: dict) -> Tuple[float, float, float, float]:
            """ 獲取 [民調單位向量各分量] 以及和選舉結果的 [餘弦相似度]

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
            survey_support_rate_uarr = survey_support_rate_arr / \
                np.linalg.norm(survey_support_rate_arr)

            cos_similarity = np.dot(
                survey_support_rate_uarr, cls.get_result_support_rate_uarr()
            )

            return (
                *survey_support_rate_uarr.tolist(),
                cos_similarity,
            )

        # 獲取委託調查單位: 去除空白、後綴數字、圓括號附註內容
        raw_df = cls.get_raw_df()
        df = pd.DataFrame()
        df["ORG"] = raw_df[raw_df.columns[0]].map(
            lambda unit_str: (
                re.sub(
                    r"\d+$",
                    "",
                    (
                        unit_str.split("（")[0] if "（" in unit_str else unit_str
                    )
                    .replace(" ", "")
                    # 處理同單位變名
                    .replace("東森新聞", "東森")
                )
            )
        )

        # 獲取調查結束時間
        df["survey_date"] = pd.to_datetime(
            raw_df[raw_df.columns[1]].map(cls.formate_survey_date_str)
        )

        # 獲取調查結束時間離選舉的天數
        df["survey_date_countdown_days"] = df["survey_date"].map(
            lambda date: (
                cls.result_date - date.date()
            ).days
        )

        # 計算各候選人站票比例 以及 [餘弦相似度]
        *survey_rate_uarr_list, df["cos_similarity"] = zip(
            *raw_df.apply(
                get_survey_rate_uarr_and_cos_similarity,
                axis=1,
            )
        )
        for person_name, survey_rate_uarr in zip(
            cls.result_support_rate_dict.keys(), survey_rate_uarr_list
        ):
            df[person_name] = survey_rate_uarr

        # 根據 [餘弦相似度] 排序資料
        df.sort_values(by="cos_similarity", ascending=False, inplace=True)

        # 排除重複的調查單位，只留下該單位多次的民調中，[餘弦相似度] 最高的民調結果
        df.drop_duplicates(
            subset=["ORG"],
            inplace=True,
        )

        return df

    @classmethod
    def plot_ternary(cls):

        df = cls.get_df()

        fig = px.scatter_ternary(
            df,
            a=cls.get_person_name_list()[0],
            b=cls.get_person_name_list()[1],
            c=cls.get_person_name_list()[2],
            text="ORG",
            size=df["survey_date_countdown_days"],
        )

        fig.add_trace(
            go.Scatterternary(
                a=[cls.get_result_support_rate_uarr()[0]],
                b=[cls.get_result_support_rate_uarr()[1]],
                c=[cls.get_result_support_rate_uarr()[2]],
                marker=dict(
                    symbol="star",
                    size=20,
                ),
                text="選舉結果",
            )
        )

        fig.update_traces(textposition='bottom center')

        fig.show(
            config={
                'scrollZoom': True,
            }
        )

    @classmethod
    def plot_scatter(cls):
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
            y="cos_similarity",
            text="ORG",
            size=df["cos_similarity"].map(lambda x: random.random()*10),
            size_max=60,
        )

        fig.add_trace(
            go.Scatter(
                x=[result_person_a_support_rate, result_person_a_support_rate],
                y=[df['cos_similarity'].min(), 1.005],
                mode='lines',
                line=dict(
                    color='rgb(255, 0, 0)',
                    width=2,
                    dash="dash",
                ),
            )
        )
        fig.update_traces(textposition='top center')

        fig.update_layout(
            title='person_a_support_rate vs cos_similarity',
            xaxis_title=f'{person_name_a} 佔票比例',
            yaxis_title='cos_similarity',
        )
        fig.show()


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


class ElectionPollsPresident2020(ElectionPolls):
    """ 2020年中華民國總統選舉民意調查 
    """
    url = "https://zh.wikipedia.org/wiki/2020年中華民國總統選舉民意調查#英德配－國政配－瑜湘配"
    table_index = 0
    result_date = datetime.date(2020, 1, 11)
    result_support_rate_dict = {
        "蔡英文": 57.13,
        "韓國瑜": 38.61,
        "宋楚瑜": 4.25,
    }

    @classmethod
    def formate_survey_date_str(cls, survey_date_str: str) -> str:
        _, survey_date_str = survey_date_str.split("－")
        survey_date_m, survey_date_d = survey_date_str.split("-")
        return f"2019-{survey_date_m}-{survey_date_d}"


class ElectionPollsTaipei2018(ElectionPolls):
    """ 2018年臺北市市長選舉民意調查
    """
    url = "https://zh.wikipedia.org/wiki/2018年中華民國直轄市長及縣市長選舉民意調查#_臺北市"
    table_index = 2
    result_date = datetime.date(2018, 11, 24)
    result_support_rate_dict = {
        "柯文哲": 41.06,
        "丁守中": 40.81,
        "姚文智": 17.28,
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
