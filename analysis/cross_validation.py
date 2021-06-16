import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


class CrossValidationStrategy():
    """Abstract CrosssValidationStrategy"""
    
    _nb_folds: int
    _nb_cross_validations: int
    _data: pd.DataFrame
    _random_state = None
    _cross_validations: list # list of dict('train', 'test')
    
    def __init__(
            self, 
            data: pd.DataFrame,
            nb_folds: int,
            nb_cross_validations: int,
            random_state = None
            ):
        self._data = data
        self._nb_folds = nb_folds
        self._nb_cross_validations = nb_cross_validations
        self._random_state = random_state
        self._cross_validations = []
    
    @property
    def nb_folds(self) -> int:
        return self._nb_folds 
    
    @property
    def nb_cross_validations(self) -> int:
        return self._nb_cross_validations
    
    @property
    def cross_validations(self) -> list:
        return self._cross_validations
    
    def _generate_train_test(self, train_index, test_index) -> dict:
        test_samples = list(self._data.iloc[test_index].index)
        train_samples = list(self._data.iloc[train_index].index)
        dict_train_test = dict()
        dict_train_test['train'] = train_samples
        dict_train_test['test'] = test_samples
        return dict_train_test
    
    def generate_cross_validations(self):
        pass

    def __str__(self):
        return 'Abstract CrossValidationStrategy'



class KFoldStrategy(CrossValidationStrategy):
    
    def generate_cross_validations(self):
        self._cross_validations.clear()
        cv = RepeatedKFold(n_splits=self._nb_folds, n_repeats=self._nb_cross_validations, random_state=self._random_state)
        for train_index, test_index in cv.split(self._data):
            dict_train_test = self._generate_train_test(train_index, test_index)
            self._cross_validations.append(dict_train_test)


    def __str__(self):
        return 'KFoldStrategy'
 
            
class StratifiedKFoldStrategy(CrossValidationStrategy):
    
    _targets: pd.Series
            
    def __init__(
            self, 
            data: pd.DataFrame,
            targets: pd.Series,
            nb_folds: int,
            nb_cross_validations: int,
            random_state = None
            ):
        super().__init__(data, nb_folds, nb_cross_validations, random_state)
        self._targets = targets
        
    def generate_cross_validations(self):
        self._cross_validations.clear()
        cv = RepeatedStratifiedKFold(n_splits=self._nb_folds, n_repeats=self._nb_cross_validations, random_state=self._random_state)
        for train_index, test_index in cv.split(self._data, self._targets):
            dict_train_test = self._generate_train_test(train_index, test_index)
            self._cross_validations.append(dict_train_test)
            
    def __str__(self):
        return 'StratifiedKFoldStrategy'
