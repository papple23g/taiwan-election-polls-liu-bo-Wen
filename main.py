from models import (
    ElectionPollsHsinchuCity2022,
    ElectionPollsKaohsiung2018,
    ElectionPollsNewTaipei2018,
    ElectionPollsNewTaipei2022,
    ElectionPollsPresident2012,
    ElectionPollsPresident2016,
    ElectionPollsPresident2020,
    ElectionPollsTaiChung2018,
    ElectionPollsTainan2018,
    ElectionPollsTaipei2018,
    ElectionPollsTaipeiCity2022,
    ElectionPollsTaoyuan2018,
    ElectionPollsTaoyuan2022,
    plot_survey_ranking_table,
)

if __name__ == '__main__':
    # 繪製某次選舉的民調資料表 (三元圖/二元點陣圖、民調機構排名表)
    (
        # ElectionPollsPresident2012
        # ElectionPollsPresident2016
        # ElectionPollsTaipei2018
        # ElectionPollsNewTaipei2018
        # ElectionPollsTaoyuan2018
        # ElectionPollsTaiChung2018
        # ElectionPollsTainan2018
        # ElectionPollsKaohsiung2018
        # ElectionPollsPresident2020
        ElectionPollsHsinchuCity2022
        # ElectionPollsTaipeiCity2022
        # ElectionPollsNewTaipei2022
        # ElectionPollsTaoyuan2022
    ).plot()

    # # 繪製民調單位排名資料表
    # plot_survey_ranking_table(
    #     by=(
    #         # 'median'  # 中位數排名
    #         'mean'  # 平均值排名
    #     )
    # )
