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
)

if __name__ == '__main__':
    # ElectionPollsPresident2012.plot_ternary()
    # ElectionPollsPresident2016.plot_ternary()
    # ElectionPollsTaipei2018.plot_ternary()
    # ElectionPollsNewTaipei2018.plot_scatter()
    # ElectionPollsTaoyuan2018.plot_scatter()
    # ElectionPollsTaiChung2018.plot_scatter()
    # ElectionPollsTainan2018.plot_scatter()
    ElectionPollsKaohsiung2018.plot_scatter()
    # ElectionPollsPresident2020.plot_ternary()

    # # debug
    # # ---------------------------------------
    # raw_df = (
    #     ElectionPollsPresident2012
    # ).get_raw_df()
    # print(raw_df)

    # df = (
    #     ElectionPollsPresident2012
    # ).get_df()
    # print(df)
