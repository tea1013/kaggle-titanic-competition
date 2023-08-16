import os
from typing import List, Union

import numpy as np
from numpy import ndarray
from ocha.common.logger import FileLogger, StdoutLogger
from ocha.common.notification import Notification
from ocha.dataset.cross_validator import CrossValidator
from ocha.experiment.experiment import Experiment, ExperimentConfig
from ocha.experiment.results import ExperimentResult, RemakeResult, TestResult, TrainResult, ValidResult
from ocha.util.timer import Timer
from pandas import DataFrame

from ..context import TitanicContext
from ..dataset import TitanicDataset
from ..metrics import Accuracy
from .lgb import LGBMBinary, LGBMBinaryConfig


class TitanicExperimentConfig(ExperimentConfig):
    def __init__(
        self,
        exp_name: str,
        version: int,
        n_fold: int,
        seed: int,
        cross_validator: CrossValidator,
        scoring: Accuracy,
        file_logger: FileLogger,
        std_logger: StdoutLogger,
        notification: Notification,
        is_debug: bool = False,
    ) -> None:
        super().__init__(
            exp_name,
            version,
            n_fold,
            seed,
            scoring,
            file_logger,
            std_logger,
            notification,
            is_debug,
        )
        self.cross_validator = cross_validator

        self.model_confs: List[str] = []
        os.makedirs(f"./oof/{exp_name}", exist_ok=True)
        os.makedirs(f"./models/{exp_name}", exist_ok=True)
        os.makedirs(f"./submission/{exp_name}", exist_ok=True)


class TitanicExperiment(Experiment):
    def __init__(self, context: TitanicContext, config: TitanicExperimentConfig, folds: List[int]) -> None:
        super().__init__(context, config, folds)

    def build_conf(self, fold: int, categorical_features: list[str]) -> LGBMBinaryConfig:
        return LGBMBinaryConfig(
            save_dir=self.config.exp_name,
            save_file_name=f"fold-{fold}",
            model_file_type="pickle",
            seed=self.config.seed,
            categorical_features=categorical_features,
            num_boost_round=10000,
            max_bin=500,
            is_debug=self.config.is_debug,
        )

    def build_model(self, conf: LGBMBinaryConfig) -> LGBMBinary:
        return LGBMBinary(
            conf,
            self.config.scoring,
            self.config.file_logger,
            self.config.std_logger,
        )

    def run(self) -> ExperimentResult:
        timer = Timer()
        timer.start()

        train_result = self.train()

        self.config.std_logger.info("Saveing oof ...")
        oof_df = self.save_oof(train_result.oof_prediction, train_result.score)
        self.config.std_logger.info("done.")

        self.config.std_logger.info("Prediction ...")
        test_result = self.test()
        self.config.std_logger.info("done.")

        self.config.std_logger.info("Saving submission_df ...")
        submission_df = self.save_submission(test_result.test_prediction, train_result.score)

        timer.end()

        self.config.notification.notify(f"Experiment End. [score: {train_result.score}, time: {timer.result}]")

        return ExperimentResult(
            fit_results=train_result.fit_results,
            oof_prediction=train_result.oof_prediction,
            test_prediction=test_result.test_prediction,
            oof_df=oof_df,
            submission_df=submission_df,
            score=train_result.score,
            time=timer.result,
        )

    def train(self) -> TrainResult:
        fit_results = []
        oof_prediction = np.zeros(len(self.context.sample_oof_df))
        for fold in self.folds:
            train_idx, valid_idx = self.config.cross_validator.fold_index(fold=fold)
            train = self.context.train.iloc[train_idx].reset_index(drop=True)
            valid = self.context.train.iloc[valid_idx].reset_index(drop=True)
            dataset = TitanicDataset(train, valid, self.context.test)
            dataset.processing_train()
            dataset.processing_valid()

            conf = self.build_conf(fold=fold, categorical_features=dataset.categorical_features)
            model = self.build_model(conf=conf)
            result = model.fit(dataset.train_X, dataset.train_y, dataset.valid_X, dataset.valid_y)

            fit_results.append(result)
            oof_prediction[valid_idx] = result.oof_prediction

        score = self.config.scoring.execute(self.context.train["Survived"].values, oof_prediction)

        return TrainResult(fit_results=fit_results, oof_prediction=oof_prediction, score=score)

    def valid(self) -> ValidResult:
        oof_prediction = np.zeros(len(self.context.train))
        for fold in self.folds:
            train_idx, valid_idx = self.config.cross_validator.fold_index(fold=fold)
            train = self.context.train.iloc[train_idx].reset_index(drop=True)
            valid = self.context.train.iloc[valid_idx].reset_index(drop=True)
            dataset = TitanicDataset(train, valid, self.context.test)
            dataset.processing_train()
            dataset.processing_valid()

            conf = self.build_conf(fold=fold, categorical_features=dataset.categorical_features)
            model = self.build_model(conf=conf)
            model.load()

            oof_prediction[valid_idx] = model.predict(dataset.valid_X)

        score = self.config.scoring.execute(self.context.train["Survived"].values, oof_prediction)

        return ValidResult(oof_prediction=oof_prediction, score=score)

    def test(self) -> TestResult:
        predictions = []
        for fold in self.folds:
            train_idx, valid_idx = self.config.cross_validator.fold_index(fold=fold)
            train = self.context.train.iloc[train_idx].reset_index(drop=True)
            valid = self.context.train.iloc[valid_idx].reset_index(drop=True)
            dataset = TitanicDataset(train, valid, self.context.test)
            dataset.processing_train()
            dataset.processing_test()

            conf = self.build_conf(fold=fold, categorical_features=dataset.categorical_features)
            model = self.build_model(conf=conf)
            model.load()

            prediction = model.predict(dataset.test_X)
            predictions.append(prediction)

        test_prediction = np.mean(predictions, axis=0)

        return TestResult(test_prediction=test_prediction)

    def test_seq(self, test_X: Union[DataFrame, ndarray]) -> TestResult:
        pass

    def optimize(self) -> None:
        pass

    def save_oof(self, oof_prediction: ndarray, score: float) -> DataFrame:
        oof_df = self.context.make_oof(oof_prediction)
        oof_df.to_csv(
            f"./oof/{self.config.exp_name}/tea_v{self.config.version}_oof_cv{score:.4f}.csv",
            index=False,
        )

        return oof_df

    def save_submission(self, test_prediction: ndarray, score: float) -> DataFrame:
        submission_df = self.context.make_submission(test_prediction)
        submission_df.to_csv(
            f"./submission/{self.config.exp_name}/tea_v{self.config.version}_submission_cv{score:.4f}.csv",
            index=False,
        )

        return submission_df

    def remake(self) -> RemakeResult:
        valid_result = self.valid()
        test_result = self.test()

        oof_df = self.save_oof(valid_result.oof_prediction, valid_result.score)
        submission_df = self.save_submission(test_result.test_prediction, valid_result.score)

        return RemakeResult(oof_df=oof_df, submission_df=submission_df)
