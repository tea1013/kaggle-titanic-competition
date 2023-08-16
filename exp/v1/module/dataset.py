from ocha.dataset.dataset import Dataset
from pandas import DataFrame


class TitanicDataset(Dataset):
    def __init__(self, train: DataFrame, valid: DataFrame, test: DataFrame) -> None:
        self.train = train
        self.valid = valid
        self.test = test

    @property
    def all_features(self) -> list[str]:
        return ["Pclass_encoded", "Sex_encoded", "Embarked_encoded", "Cabin_encoded", "Age", "SibSp", "Parch"]

    @property
    def categorical_features(self) -> list[str]:
        return ["Pclass_encoded", "Sex_encoded", "Embarked_encoded", "Cabin_encoded"]

    @property
    def continuous_features(self) -> list[str]:
        result = []
        for col in self.all_features:
            if col not in self.categorical_features:
                result.append(col)

        return result

    @property
    def targets(self) -> list[str]:
        return "Survived"

    def processing_train(self) -> None:
        self.train_X = self.train[self.all_features]
        self.train_y = self.train[self.targets]

    def processing_valid(self) -> None:
        self.valid_X = self.valid[self.all_features]
        self.valid_y = self.valid[self.targets]

    def processing_test(self) -> None:
        self.test_X = self.test[self.all_features]
