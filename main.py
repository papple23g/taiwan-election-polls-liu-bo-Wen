from models import (
    ElectionPollsKaohsiung2018,
    ElectionPollsPresident2016,
    ElectionPollsPresident2020,
    ElectionPollsTaiChung2018,
    ElectionPollsTaipei2018,
    ElectionPollsNewTaipei2018,
    ElectionPollsKaohsiung2018,
    ElectionPollsTaoyuan2018,
)

if __name__ == '__main__':
    # ElectionPollsPresident2016.plot_ternary()
    # ElectionPollsPresident2020.plot_ternary()
    # ElectionPollsTaipei2018.plot_ternary()
    # ElectionPollsNewTaipei2018.plot_scatter()
    # ElectionPollsTaoyuan2018.plot_scatter()
    # ElectionPollsTaiChung2018.plot_scatter()

    ElectionPollsKaohsiung2018.plot_scatter()

    # # debug
    # # ---------------------------------------
    # raw_df = (
    #     # ElectionPollsKaohsiung2018
    #     # ElectionPollsPresident2016
    #     # ElectionPollsPresident2020
    #     # ElectionPollsTaipei2018
    #     # ElectionPollsNewTaipei2018
    #     # ElectionPollsKaohsiung2018
    #     ElectionPollsTaiChung2018
    # ).get_raw_df()
    # print(raw_df)

    # df = (
    #     # ElectionPollsTaipei2018
    #     # ElectionPollsKaohsiung2018
    #     ElectionPollsTaiChung2018
    # ).get_df()
    # print(df)
