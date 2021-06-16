import pandas as pd

class ExpressionFrequency:
    
    def define_expressed_samples(self, data: pd.Series, threshold: pd.Series) -> pd.DataFrame:
        return data > threshold
    
    def calculate_expression_frequency(self, data: pd.Series, threshold: pd.Series):
        expressed = self.define_expressed_samples(data, threshold)
        return 100.0 * expressed.sum() / expressed.shape[0]

        