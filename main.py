from models import (
    ElectionPollsPresident2016,
    ElectionPollsPresident2020,
    ElectionPollsTaipei2018,
    ElectionPollsNewTaipei2018,
)

import pandas as pd
import pylab as plt

if __name__ == '__main__':
    # ElectionPollsPresident2020.plot_ternary()
    # ElectionPollsPresident2016.plot_ternary()
    # ElectionPollsTaipei2018.plot_ternary()

    # raw_df = ElectionPollsNewTaipei2018.get_raw_df()
    # print(raw_df)

    # df = ElectionPollsNewTaipei2018.get_df()
    ElectionPollsNewTaipei2018.plot_scatter()
