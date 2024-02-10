# Conformal Impact
Take Causal Impact and replace the Bayesian Structural Time Series Model with MFLES and the Basyesian posterior with Conformal Prediction Intervals.

## Quick Examnple an comparison to Causal Impact
```
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


from ConformalImpact.Model import CI


conformal_impact = CI(opt_size=20,
                      opt_steps=10,
                      opt_step_size=3)
impact_df = conformal_impact.fit(data,
                              n_windows=30,
                              intervention_index=100,
                              seasonal_period=None)

conformal_impact.summary()
conformal_impact.plot()





from causalimpact import CausalImpact

impact = CausalImpact(data, [0, 99], [100, 130])
impact.run()
impact.plot()
print(impact.summary())
output = impact.inferences
np.mean(output['point_effect'].values[-30:])
```