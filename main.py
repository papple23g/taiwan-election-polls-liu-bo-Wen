import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import requests

# get DataFrame from url


def get_html_str(url: str) -> str:
    html_path = Path(__file__)/'../00.html'

    if html_path.exists():
        with html_path.open('r', encoding='utf8') as f:
            return f.read()

    with html_path.open('w', encoding='utf8') as f:
        response = requests.get(url)
        f.write(response.text)
    return response.text


# 獲取表格: 2020年中華民國總統選舉民意調查（英德配－國政配－瑜湘配）
url = "https://zh.wikipedia.org/wiki/2020%E5%B9%B4%E4%B8%AD%E8%8F%AF%E6%B0%91%E5%9C%8B%E7%B8%BD%E7%B5%B1%E9%81%B8%E8%88%89%E6%B0%91%E6%84%8F%E8%AA%BF%E6%9F%A5#%E8%94%A1%E8%8B%B1%E6%96%87%EF%BC%8D%E9%9F%93%E5%9C%8B%E7%91%9C_2"
html_str = get_html_str(url)
df_list: List[pd.DataFrame] = pd.read_html(html_str)
df = df_list[0]

# 重整欄位列表
df.columns = [cols[-1] for cols in df.columns]

# 僅擷取全國民調
df: pd.DataFrame = df.iloc[
    :df[df["有效樣本"].str.contains("市")].first_valid_index(),
]

# 去除 [有效樣本] 欄位
df.drop(['有效樣本'], inplace=True, axis=1)

# 簡化 [委託調查單位] 欄位
df["委託調查單位"] = df["委託調查單位"].map(
    lambda unit_str: (
        unit_str.split("（")[0] if "（" in unit_str else unit_str
    )
)


def get_cos_similarity_and_prefs(row_dict: dict) -> Tuple[float, float, float, float]:
    """ 計算與實際投票結果的餘弦相似度，以及偏移分量

    Returns:
        float: 餘弦相似度
    """

    gbo_result_arr = np.array([
        57.13,
        38.61,
        4.25,
    ])

    vote_num_arr = np.array([
        float(row_dict[key].replace("%", ""))
        for key in row_dict.keys()
        if any(
            kw in key for kw in [
                "蔡英文",
                "韓國瑜",
                "宋楚瑜",
            ]
        )
    ])

    # 獲取餘弦相似度
    cos_similarity = np.dot(gbo_result_arr, vote_num_arr) / (
        np.linalg.norm(gbo_result_arr) * np.linalg.norm(vote_num_arr)
    )

    # 獲取偏移分量
    vote_num_uarr: np.ndarray = vote_num_arr / np.linalg.norm(vote_num_arr)
    gbo_result_uarr: np.ndarray = gbo_result_arr / \
        np.linalg.norm(gbo_result_arr)
    perf_uarr: np.ndarray = (vote_num_uarr - gbo_result_uarr)

    return (
        cos_similarity,
        *perf_uarr.tolist(),
    )


df["餘弦相似度"], df["pref_g"], df["pref_b"], df["pref_o"] = zip(
    *df.apply(
        get_cos_similarity_and_prefs,
        axis=1,
    )
)
df.sort_values(by="餘弦相似度", ascending=False, inplace=True)

print(df)
print()

# def
# print(set(df["委託調查單位"]))


# a_arr = np.array([
#     -10, -40, -3
# ])

# cos_similarity = np.dot(gbo_result_arr, a_arr) / \
#     (np.linalg.norm(gbo_result_arr) * np.linalg.norm(a_arr))
# print(cos_similarity)
