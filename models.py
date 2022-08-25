import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import requests


class ElectionPollsPresident2020:
    url = "https://zh.wikipedia.org/wiki/2020%E5%B9%B4%E4%B8%AD%E8%8F%AF%E6%B0%91%E5%9C%8B%E7%B8%BD%E7%B5%B1%E9%81%B8%E8%88%89%E6%B0%91%E6%84%8F%E8%AA%BF%E6%9F%A5#%E8%94%A1%E8%8B%B1%E6%96%87%EF%BC%8D%E9%9F%93%E5%9C%8B%E7%91%9C_2"
    table_index = 0
    result_support_rate_dict = {
        "蔡英文": 57.13,
        "韓國瑜": 38.61,
        "宋楚瑜": 4.25,
    }

    @classmethod
    def get_person_name_list(cls):
        return list(cls.result_support_rate_dict.keys())

    @classmethod
    def get_result_support_rate_uarr(cls):
        result_support_rate_arr = np.array(
            list(cls.result_support_rate_dict.values())
        )
        return result_support_rate_arr / (
            np.linalg.norm(result_support_rate_arr)
        )

    @classmethod
    def get_html_str(cls) -> str:
        html_path = Path(__file__)/f'../html/{cls.__name__}.html'

        if html_path.exists():
            with html_path.open('r', encoding='utf8') as f:
                return f.read()

        with html_path.open('w', encoding='utf8') as f:
            response = requests.get(cls.url)
            f.write(response.text)
        return response.text

    @classmethod
    def get_raw_df(cls):
        html_str = cls.get_html_str()
        raw_df_list: List[pd.DataFrame] = pd.read_html(html_str)
        raw_df = raw_df_list[cls.table_index]

        # 重整欄位列表
        raw_df.columns = [cols[-1] for cols in raw_df.columns]

        # 僅擷取全國民調
        raw_df: pd.DataFrame = raw_df.iloc[
            :raw_df[
                ~raw_df[raw_df.columns[-1]].str.contains("%")
            ].first_valid_index(),
        ]

        return raw_df

    @classmethod
    def get_df(cls):

        raw_df = cls.get_raw_df()
        df = pd.DataFrame()
        df["ORG"] = raw_df["委託調查單位"].map(
            lambda unit_str: (
                unit_str.split("（")[0] if "（" in unit_str else unit_str
            )
        )

        def get_predict_rate_uarr_and_cos_similarity(row_dict: dict) -> Tuple[float, float, float, float]:
            """ 獲取 [民調單位向量各分量] 以及和選舉結果的 [餘弦相似度]

            Returns:
                Tuple[float, float, float, float]
            """
            predict_support_rate_arr = np.array([
                float(row_dict[key].replace("%", ""))
                for key in row_dict.keys()
                if any(
                    kw in key for kw in cls.result_support_rate_dict.keys()
                )
            ])
            predict_support_rate_uarr = predict_support_rate_arr / \
                np.linalg.norm(predict_support_rate_arr)

            cos_similarity = np.dot(
                predict_support_rate_uarr, cls.get_result_support_rate_uarr()
            )

            return (
                *predict_support_rate_uarr.tolist(),
                cos_similarity,
            )

        # 計算各候選人站票比例 以及 [餘弦相似度]
        # df["g_ratio"], df["b_ratio"], df["o_ratio"], df["cos_similarity"] = zip(
        *predict_rate_uarr_list, df["cos_similarity"] = zip(
            *raw_df.apply(
                get_predict_rate_uarr_and_cos_similarity,
                axis=1,
            )
        )
        for person_name, predict_rate_uarr in zip(
            cls.result_support_rate_dict.keys(), predict_rate_uarr_list
        ):
            df[person_name] = predict_rate_uarr

        # 根據 [餘弦相似度] 進行排序
        df.sort_values(by="cos_similarity", ascending=False, inplace=True)

        # 排除重複的調查單位，只留下該單位 [餘弦相似度] 最高的民調結果
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
        )

        fig.add_trace(
            go.Scatterternary(
                a=[cls.get_result_support_rate_uarr()[0]],
                b=[cls.get_result_support_rate_uarr()[1]],
                c=[cls.get_result_support_rate_uarr()[2]],
                marker=dict(
                    symbol="star",
                ),
                text="選舉結果",
                showlegend=False,
                textposition="bottom center",
            )
        )

        fig.update_traces(textposition='top center')

        fig.show(
            config={
                # "modeBarButtons": "modeBarButtons"
                "displayModeBar": True,
            }
        )


ElectionPollsPresident2020.plot_ternary()
