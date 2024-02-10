# -*- coding: utf-8 -*-
from MFLES.Forecaster import MFLES
import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
sns.set_style('darkgrid')


class CI():

    def __init__(self,
                 opt_size,
                 opt_steps,
                 opt_step_size,
                 ):
        self.opt_size = opt_size
        self.opt_steps = opt_steps
        self.opt_step_size = opt_step_size

    def fit(self, data, intervention_index, n_windows=30, fit_intercept=True, seasonal_period=None):
        data = data.copy()
        index = data.index
        intervention = list(index).index(intervention_index)
        data = data.reset_index()

        y = data['y'].values
        X = data.drop('y', axis=1)
        if fit_intercept:
            X['constant'] = 1
        X = X.values
        train_X, test_X = X[:intervention], X[intervention:]
        train_y, test_y = y[:intervention], y[intervention:]

        configs = {
            'X': [train_X],
            'seasonality_weights': [True, False],
            'multiplicative': [True, False],
            'smoother': [True, False],
            'max_rounds': [3, 5, 7, 10, 20],
            'exogenous_lr': [.2, .5, .7, .9, 1],
            'seasonal_lr': [0, .2, .5, .7, .9, 1],
            'seasonal_period': [seasonal_period],
            }
        self.model = MFLES()
        self.opt_params = self.model.optimize(train_y,
                                    test_size=self.opt_size,
                                    n_steps=self.opt_steps,
                                    step_size=self.opt_step_size,
                                    params=configs)
        point_estimates, upper_bound, lower_bound = self.model.conformal(train_y,
                                                                    forecast_horizon=len(test_y),
                                                                    n_windows=n_windows,
                                                                    coverage=[.9,.95,.99],
                                                                    future_X=test_X,
                                                                    **self.opt_params)
        self.upper_bounds = upper_bound
        self.lower_bounds = lower_bound

        pos_resids = test_y - upper_bound[0][-len(test_y):]
        neg_resids = test_y - lower_bound[0][-len(test_y):]
        pos_resids = np.clip(pos_resids, a_max=None, a_min=0)
        neg_resids = np.clip(neg_resids, a_max=0, a_min=None)

        impact = pos_resids + neg_resids

        data['yhat'] = point_estimates
        data['upper_bound'] = upper_bound[0]
        data['lower_bound'] = lower_bound[0]
        data['pointwise'] = np.append(np.zeros(len(train_y)), impact)
        data['cumulative_impact'] = np.cumsum(np.append(np.zeros(len(train_y)), impact))
        data['average_impact'] = np.mean(impact)
        data['residuals'] = data['y'] - data['yhat']
        data['intervention_index'] = intervention_index
        self.impact_ci = st.norm.interval(alpha=0.95, loc=np.mean(impact), scale=st.sem(impact))
        self.data = data
        self.impact = impact
        return data

    def plot(self):
        fig, ax = plt.subplots(3)
        ax[0].plot(self.data['y'], color='black')
        ax[0].axvline(self.data['intervention_index'].iloc[0], color='black', linestyle='dashed')
        ax[0].plot(self.data['yhat'], color='red', linestyle='dashed')
        ax[0].fill_between(self.data.index,
                         self.data['upper_bound'],
                         self.data['lower_bound'], alpha=.2,color='black')
        ax[1].plot(self.data['residuals'], color='red', linestyle='dashed')
        ax[1].axhline(0, color='green')
        ax[1].axvline(self.data['intervention_index'].iloc[0], color='black', linestyle='dashed')
        ax[2].plot(self.data['cumulative_impact'], color='red', linestyle='dashed')
        ax[2].axhline(0, color='green')
        ax[2].axvline(self.data['intervention_index'].iloc[0], color='black', linestyle='dashed')
        fig.tight_layout()
        plt.show()

    def summary(self):
        print(f'AVG. Absolute Effect: {np.mean(self.impact)}')
        print(f'Cumulative Effect: {np.sum(self.impact)}')



#%%
if __name__ == '__main__':
    intervention_effect = 400
    np.random.seed(42)
    series = np.random.random((130, 1)) * 400
    x_series = series * .4 + np.random.random((130, 1)) * 50 + 1000
    trend = (np.arange(1, 131)).reshape((-1, 1))
    series += 10 * trend
    series[-30:] = series[-30:] + intervention_effect

    data = pd.DataFrame(np.column_stack([series, x_series]), columns=['y', 'x1'])

    import matplotlib.pyplot as plt

    plt.plot(series)
    plt.plot(x_series)
    plt.show()

    conformal_impact = CI(opt_size=20,
                          opt_steps=10,
                          opt_step_size=3)
    impact_df = conformal_impact.fit(data,
                                  n_windows=30,
                                  intervention_index=100,
                                  seasonal_period=None)

    conformal_impact.summary()
    conformal_impact.plot()

    opt_params = conformal_impact.opt_params

    from causalimpact import CausalImpact

    impact = CausalImpact(data, [0, 99], [100, 130])
    impact.run()
    impact.plot()
    print(impact.summary())
    output = impact.inferences
    np.mean(output['point_effect'].values[-30:])

