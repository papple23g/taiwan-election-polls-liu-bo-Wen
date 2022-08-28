from models import (
    ElectionPollsPresident2012,
    ElectionPollsKaohsiung2018,
    ElectionPollsPresident2016,
    ElectionPollsPresident2020,
    ElectionPollsTaiChung2018,
    ElectionPollsTainan2018,
    ElectionPollsTaipei2018,
    ElectionPollsNewTaipei2018,
    ElectionPollsKaohsiung2018,
    ElectionPollsTaoyuan2018,
    plot_survey_ranking_table,
)

if __name__ == '__main__':
    # # 繪製某次選舉的圖表
    # ElectionPollsPresident2012.plot_ternary()
    # ElectionPollsPresident2016.plot_ternary()
    # ElectionPollsTaipei2018.plot_ternary()
    # ElectionPollsNewTaipei2018.plot_scatter()
    # ElectionPollsTaoyuan2018.plot_scatter()
    # ElectionPollsTaiChung2018.plot_scatter()
    # ElectionPollsTainan2018.plot_scatter()
    # ElectionPollsKaohsiung2018.plot_scatter()
    ElectionPollsPresident2020.plot_ternary()

    # 繪製某次選舉的民調資料表
    (
        # ElectionPollsPresident2012
        # ElectionPollsPresident2016
        # ElectionPollsTaipei2018
        # ElectionPollsNewTaipei2018
        # ElectionPollsTaoyuan2018
        # ElectionPollsTaiChung2018
        # ElectionPollsTainan2018
        # ElectionPollsKaohsiung2018
        ElectionPollsPresident2020
    ).plot_survey_table()

    # # 繪製民調單位排名資料表
    # plot_survey_ranking_table(
    #     by=(
    #         'median'  # 中位數排名
    #         # 'mean' # 平均值排名
    #     )
    # )
