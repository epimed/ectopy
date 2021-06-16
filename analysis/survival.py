import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

class SurvivalModel:
    """Abstract survival model"""
    
    _survival_data: pd.DataFrame
    _duration_col: str
    _event_col: str
    _cph: CoxPHFitter
    
    def __init__(
            self,
            survival_data: pd.DataFrame, 
            duration_col: str = 'time', 
            event_col:str = 'event'
            ):
        self._survival_data = survival_data
        self._duration_col = duration_col
        self._event_col = event_col
        self._cph = CoxPHFitter()
        
    def calculate_binarized_follow_up(self) -> pd.Series:
        events_only = self._survival_data[self._survival_data[self._event_col]>0]
        median_follow_up = events_only[self._duration_col].median()
        over_median = self._survival_data[self._duration_col]>median_follow_up
        bin_follow_up = pd.Series(index=self._survival_data.index, dtype=float)
        bin_follow_up[(~over_median)] = 0.0
        bin_follow_up[over_median] = 1.0
        return bin_follow_up    
        
    def generate_expression_survival_data(self, feature, data: pd.DataFrame) -> pd.DataFrame:
        survival_expression = pd.DataFrame(index=data.index)
        survival_expression['feature'] = data[feature]
        survival_expression['time'] = self._survival_data.loc[survival_expression.index, self._duration_col]
        survival_expression['event'] = self._survival_data.loc[survival_expression.index, self._event_col]
        return survival_expression[['feature', 'time', 'event']]
    
    def generate_group_survival_data(self, feature, threshold, data: pd.DataFrame) -> pd.DataFrame:
        group_survival = self.generate_expression_survival_data(feature, data)
        group_survival['group'] = 0
        group_survival.loc[group_survival['feature']>threshold, 'group'] = 1
        return group_survival[['group', 'time', 'event']]
    
    def calculate_model_for_expression(self, feature, data: pd.DataFrame) -> tuple:
        cox_expression = self.generate_expression_survival_data(feature, data)
        self._cph.fit(cox_expression, duration_col='time', event_col='event', show_progress=False)
        cox_pvalue_expression = self._cph.summary.p['feature']
        cox_hr_expression = self._cph.summary['exp(coef)']['feature']
        return (cox_pvalue_expression, cox_hr_expression)
    
    def calculate_model_for_threshold(self, feature, threshold, data: pd.DataFrame) -> tuple:
        return (np.nan, np.nan)
    
    def is_significant(self, model_output) -> bool:
        pvalue_max = 0.05
        hr_min = 1.0
        if model_output is None:
            return False
        if len(model_output)<1:
            return False
        if len(model_output)<2:
            return (model_output[0]<=pvalue_max)
        if len(model_output)<3:
            return (model_output[0]<=pvalue_max and model_output[1]>=hr_min)


class Cox(SurvivalModel):
    
    def calculate_model_for_threshold(self, feature, threshold, data: pd.DataFrame) -> tuple:
        cox_group = self.generate_group_survival_data(feature, threshold, data)
        self._cph.fit(cox_group, duration_col='time', event_col='event', show_progress=False)
        cox_pvalue_group = self._cph.summary.p['group']
        cox_hr_group = self._cph.summary['exp(coef)']['group']
        return (cox_pvalue_group, cox_hr_group)
    

class Logrank(SurvivalModel):
    
    def calculate_model_for_threshold(self, feature, threshold, data: pd.DataFrame) -> tuple:
        cox_group = self.generate_group_survival_data(feature, threshold, data)
        self._cph.fit(cox_group, duration_col='time', event_col='event', show_progress=False)
        cox_hr_group = self._cph.summary['exp(coef)']['group']
        logrank = multivariate_logrank_test(cox_group['time'], cox_group['group'], cox_group['event'])  
        return (logrank.p_value, cox_hr_group)
    