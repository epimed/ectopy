import pandas as pd
import numpy as np
from analysis import cross_validation, survival

# === Thresholds ===

class Threshold:
    """Abstract threshold class"""
    
    _data: pd.DataFrame
    
    def __init__(self, data: pd.DataFrame):
        self._data = data
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    def calculate_threshold(self) -> pd.Series:
        pass
    
    def get_threshold_percentile(self, expression_values: pd.Series, threshold: float) -> float:
        values = expression_values.sort_values().to_numpy()
        if (len(values)==0.0):
            return 0.0
        n = 0
        for i in range(len(values)):
            if (values[i]<=threshold):
                n = n + 1
        return 100.0 * n / len(values)


class MeanTreshold(Threshold):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
    
    def calculate_threshold(self) -> pd.Series:
        return self.data.mean()


class PercentileThreshold(Threshold):

    _percentile: float
    
    def __init__(self, data: pd.DataFrame, percentile: float):
        super().__init__(data)
        self._percentile = percentile
    
    def calculate_threshold(self) -> pd.Series:
        return self.data.quantile(self._percentile/100.0)


class NSampleThreshold(Threshold):

    _nb_samples: int

    def __init__(self, data: pd.DataFrame, nb_samples: int):
        super().__init__(data)
        self._nb_samples = nb_samples
    
    def calculate_threshold(self, ascending=True) -> pd.Series:
        threshold = pd.Series(index=self.data.columns, dtype=float)
        for col in self.data.columns:
            threshold[col] = self.data[col].sort_values(axis=0, ascending=ascending).iloc[self._nb_samples-1]
        return threshold

class NoiseThreshold(Threshold):
    
    _noise_level: float

    def __init__(self, data: pd.DataFrame, noise_level: float):
        super().__init__(data)
        self._noise_level = noise_level

    def calculate_threshold(self) -> pd.Series:
        return pd.Series(self._noise_level, index=self.data.columns)
 

class MaxTreshold(Threshold):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def calculate_threshold(self) -> pd.Series:
        return self.data.max()


# === Threshold Decorators ===

class ThresholdDecorator(Threshold):
    """Abstract threshold decorator class"""
    
    _threshold: Threshold
    
    def __init__(self, threshold: Threshold):
        self._threshold = threshold
    
    @property
    def threshold(self) -> pd.Series:
        return self._threshold
    
    def calculate_threshold(self) -> pd.Series:
        pass


class StdDecorator(ThresholdDecorator):

    _nb_std: float
    
    def __init__(self, threshold: Threshold, nb_std: float = 0.0):
        super().__init__(threshold)
        self._nb_std = nb_std
    
    @property
    def nb_std(self) -> float:
        return self._nb_std
    
    def calculate_threshold(self) -> pd.Series:
        return self.threshold.calculate_threshold() + self.nb_std * self.threshold.data.std()
    

# === Adaptive threshold ===

class AdaptiveThreshold(Threshold):
    
    _survival_data: pd.DataFrame
    _duration_col: str
    _event_col: str
    
    _noise_level: float
    _percentile: float
    _step_percentile: float
    _min_nb_samples: int
    _min_reference_threshold: pd.Series
    _nb_folds: int
    _nb_cross_validations: int
    _cv_type: str
    
    _min_threshold: pd.Series
    _max_threshold: pd.Series
    
    _eligible_features: list
    _dict_thresholds: dict # {'gene' : pd.DataFrame('thresholds', 'threshold_percentiles')} 
    _cv_strategy: cross_validation.CrossValidationStrategy
    _survival_model: survival.SurvivalModel

    
    def __init__(
            self, 
            data: pd.DataFrame,
            survival_data: pd.DataFrame, 
            duration_col: str = 'time', 
            event_col:str = 'event', 
            noise_level: float = 0.3, 
            percentile: float = 15.0, 
            step_percentile: float = 1.0, 
            min_nb_samples: int = 20, 
            min_reference_threshold: pd.Series = None,
            nb_folds: int = 3,
            nb_cross_validations: int = 1,
            cv_type: str = 'stratified_k_fold'
            ):
        
        super().__init__(data)
        self._survival_data = survival_data
        self._duration_col = duration_col
        self._event_col = event_col
        self._noise_level = noise_level
        self._percentile = percentile
        self._step_percentile = step_percentile
        self._min_nb_samples = min_nb_samples
        self._min_reference_threshold = min_reference_threshold
        self._nb_folds = nb_folds
        self._nb_cross_validations = nb_cross_validations
        self._cv_type = cv_type
    
        self._dict_thresholds = dict()
        self._calulate_min_threshold()
        self._calulate_max_threshold()
        self._define_eligible_features()
        
        self._init_cv_strategy()
        self._init_survival_model()
        
    
    @property
    def survival_data(self) -> pd.DataFrame:
        return self._survival_data
        
    @property
    def min_threshold(self) -> pd.Series:
        return self._min_threshold
    
    @property
    def max_threshold(self) -> pd.Series:
        return self._max_threshold
    
    @property
    def eligible_features(self) -> list:
        return self._eligible_features
    
    @property
    def dict_thresholds(self) -> dict:
        return self._dict_thresholds
    
    @property
    def cross_validations(self) -> list:
        return self._cv_strategy.cross_validations
    
    def get_details(self, feature) -> pd.DataFrame:
        return self._dict_thresholds[feature]
        
    
    def _init_survival_model(self):
        options = {
            'survival_data': self._survival_data, 
            'duration_col': self._duration_col,
            'event_col': self._event_col
            }
        self._survival_model = survival.Cox(**options)
    
    def _init_cv_strategy(self):
        options = {
            'data': self._data, 
            'nb_folds': self._nb_folds,
            'nb_cross_validations': self._nb_cross_validations
            }
        
        if (self._cv_type=='stratified_k_fold'):
            bin_follow_up = self._get_binarized_follow_up()
            self._cv_strategy = cross_validation.StratifiedKFoldStrategy(**options, targets=bin_follow_up)
        else:
            self._cv_strategy = cross_validation.KFoldStrategy(**options)
        self._cv_strategy.generate_cross_validations()
   
        
    def _calulate_min_threshold(self):
        list_thresholds = []
        if self._percentile is not None:
            list_thresholds.append(PercentileThreshold(self.data, self._percentile).calculate_threshold())
        if self._min_nb_samples is not None:
            list_thresholds.append(NSampleThreshold(self.data, self._min_nb_samples).calculate_threshold(ascending=True))
        if self._noise_level is not None:
            list_thresholds.append(NoiseThreshold(self.data, self._noise_level).calculate_threshold())
        if self._min_reference_threshold is not None:
            list_thresholds.append(self._min_reference_threshold)
        self._min_threshold = pd.concat(list_thresholds, axis=1).max(axis=1)

    def _calulate_max_threshold(self):
        list_thresholds = []
        if self._percentile is not None:
            list_thresholds.append(PercentileThreshold(self.data, 100.0-self._percentile).calculate_threshold())
        if self._min_nb_samples is not None:
            list_thresholds.append(NSampleThreshold(self.data, self._min_nb_samples).calculate_threshold(ascending=False))
        self._max_threshold = pd.concat(list_thresholds, axis=1).min(axis=1)
        
    def _define_eligible_features(self):
        eligible_features = self._max_threshold>=self._min_threshold
        self._eligible_features = list(eligible_features[eligible_features].index)
    
    def _generate_thresholds(self): 
        for feature in self._eligible_features:
            min_percentile = self.get_threshold_percentile(self.data[feature], self.min_threshold[feature])
            max_percentile = self.get_threshold_percentile(self.data[feature], self.max_threshold[feature])
            threshold_percentiles = np.arange(min_percentile, max_percentile + self._step_percentile, self._step_percentile)
            thresholds = np.array([np.percentile(self.data[feature], p) for p in threshold_percentiles])
            self._dict_thresholds[feature] = pd.DataFrame()
            self._dict_thresholds[feature]['threshold'] = thresholds
            self._dict_thresholds[feature]['threshold_percentile'] = threshold_percentiles
            self._dict_thresholds[feature].index = ['T' + str(i+1) for i in range(len(thresholds))]
     
    def _calculate_threshold_status(self, feature):
        pvalues = []
        hrs = []
        validated = []
        for current_threshold in self.dict_thresholds[feature]['threshold']:
            model_output = self._survival_model.calculate_model_for_threshold(feature, current_threshold, self.data)
            pvalues.append(model_output[0])
            hrs.append(model_output[1])
            cox_group_validated = self._survival_model.is_significant(model_output)
            validated.append(cox_group_validated)
        self._dict_thresholds[feature]['p_value'] = pvalues
        self._dict_thresholds[feature]['hazard_ratio'] = hrs
        self._dict_thresholds[feature]['validated'] = validated  
        self._dict_thresholds[feature]['cv_score'] = np.nan
        self._dict_thresholds[feature]['optimal'] = False    
        
    def _get_candidate_thresholds(self, feature) -> pd.DataFrame:
        threshold_data = self._dict_thresholds[feature]
        return threshold_data[threshold_data['validated']==True]
    
    def _get_binarized_follow_up(self) -> pd.Series:
        events_only = self._survival_data[self._survival_data[self._event_col]>0]
        median_follow_up = events_only[self._duration_col].median()
        over_median = self._survival_data[self._duration_col]>median_follow_up
        bin_follow_up = pd.Series(index=self._survival_data.index, dtype=float)
        bin_follow_up[(~over_median)] = 0.0
        bin_follow_up[over_median] = 1.0
        return bin_follow_up
    
     
    def _calculate_cross_validation_score(self, feature):
        candidate_thresholds = self._get_candidate_thresholds(feature)
        for ind_threshold in candidate_thresholds.index:
            candidate_threshold_percentile = candidate_thresholds.loc[ind_threshold, 'threshold_percentile']
            nb_cv_validated = 0
            for ind_cv in range(len(self.cross_validations)):
                is_cv_validated = True
                dict_cross_validation_samples = self.cross_validations[ind_cv]
                for dataset in ['train', 'test']:
                    samples = dict_cross_validation_samples[dataset]
                    dataset_data = self.data.loc[samples]
                    candidate_threshold = dataset_data[feature].quantile(candidate_threshold_percentile / 100.0)
                    model_output = self._survival_model.calculate_model_for_threshold(feature, candidate_threshold, dataset_data)
                    cox_group_validated = self._survival_model.is_significant(model_output)
                    is_cv_validated = is_cv_validated and cox_group_validated
                if (is_cv_validated):
                    nb_cv_validated = nb_cv_validated + 1
            cv_score = 100.0 * nb_cv_validated / len(self.cross_validations)
            self._dict_thresholds[feature].loc[ind_threshold, 'cv_score'] = cv_score   
   
    def _get_optimal_threshold(self, feature):
        optimal = self._dict_thresholds[feature].sort_values(by=['validated', 'cv_score', 'threshold_percentile', 'p_value'], ascending=[False, False, False, True]).head(1)
        self._dict_thresholds[feature].loc[optimal.index, 'optimal'] = True
        optimal.loc[:, 'optimal'] = True
        return optimal
    
    def calculate_threshold(self) -> pd.Series:
        adaptive = pd.Series(index=self.data.columns, dtype=float)
        self._generate_thresholds()
        for feature in self._eligible_features:
            print('Processing feature', feature)
            self._calculate_threshold_status(feature)
            self._calculate_cross_validation_score(feature)
            optimal_threshold = self._get_optimal_threshold(feature)
            print('Optimal threshold for', feature)
            print(optimal_threshold)
            adaptive.loc[feature] =  optimal_threshold.iloc[0]['threshold']
        return adaptive