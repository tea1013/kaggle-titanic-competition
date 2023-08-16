import numpy as np
from numpy import ndarray
from ocha.experiment.context import Context
from pandas import DataFrame


class TitanicContext(Context):
    def __init__(
        self, train: DataFrame, test: DataFrame | None, sample_oof_df: DataFrame, sample_submission_df: DataFrame
    ) -> None:
        super().__init__(train, test, sample_oof_df, sample_submission_df)

    def make_oof(self, oof_prediction: ndarray) -> DataFrame:
        oof_df = self.sample_oof_df
        oof_df["pred"] = np.where(oof_prediction < 0.5, 0, 1)
        oof_df["prob"] = oof_prediction

        return oof_df

    def make_submission(self, test_prediction: ndarray) -> DataFrame:
        submission_df = self.sample_submission_df
        submission_df["Survived"] = np.where(test_prediction < 0.5, 0, 1)

        return submission_df
