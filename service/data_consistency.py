import pandas as pd

class DataConsistency:
      
    def check_sample_id (self, data: pd.DataFrame, expgroup: pd.DataFrame):
        if 'id_sample' in expgroup.columns:
            expgroup.index = expgroup['id_sample']
            expgroup = expgroup.drop(columns = ['id_sample'])

        data.index = data['gene_symbol']
        data = data.drop(columns=['gene_symbol'])
        
        expgroup=expgroup.dropna()
        data=data.dropna()
        common_samples = list(set(expgroup.index).intersection(set(data.columns)))
        expgroup = expgroup.loc[common_samples,:]
        data = data[common_samples]
        
        return data, expgroup

    def calculate_consistency_stats(self, data: pd.DataFrame, expgroup: pd.DataFrame) -> dict:
        stats = dict()
        common_samples = list(set(expgroup.index).intersection(set(data.index)))
        stats['expgroup'] = dict() 
        stats['data'] = dict() 
        stats['expgroup']['n_samples'] = expgroup.shape[0]
        stats['data']['n_samples'] = data.shape[0]
        stats['n_common_samples'] = len(common_samples)
        stats['common_samples'] = common_samples
        return stats
