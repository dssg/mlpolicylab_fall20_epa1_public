## Validation

### Model Selection

#### Q. How to use the `modelSelector`?

Using the output `.csv` file from training, we can run the model selector as follows:

```
m = modelSelector() # instantiate class
m.read_result_from_file([FILE TO TRAINING OUTPUT]) # read in model training result
m.parse_precision_support() # parse precision and support

m.filter_models(method=[FILTER METHOD], cutoff=[CUTOFF]) # filter models by method

ordered_models = m.rank_models(method=[RANKING METHOD]) # rank models by method

# save top models config to file for retraining
m.take_N_and_export([WITH HISTORY?], ordered_models, n=[NUM MODELS TO EXPORT], file_path=[FILE PATH TO EXPORTED MODEL CONFIG], append=[APPEND TO EXISTING FILE?])
```


#### Ranking Methods

|   | Method Name |                Description |
|:--:|:------------|:---------------------------|
| 1 | `avg_precision_overtime` | rank in descending order by the average precision over all validation sets. If tied, ranked by precision in 2014, 2013, etc. It's the **default** method. |
| 2 | `twice_as_important_overtime` | rank in descending order by the discounted average precision over all validation sets. Precision in 2014 is weighted twice as much as the precision in 2013, and so on.  |



#### Filter Methods

|   | Method Name |                Description |
|:--:|:------------|:---------------------------|
| 1 | `avg_support_overtime` | filter out the models that have average support lower than the given cutoff.  It's the **default** method. |


### Bias and Fairness Audit

#### Q. How to use the `BiasFairness` class to perform an auditing?

Simply instantiate an object as follows:

```
bf = BiasFairness(with_history=[WITH HISTORY?], metric_name=[METRIC/FEATURE NAME], ref_group=[REF GROUP], base_dir=[DIRECTORY FOR MODELS])
bf.get_bias_fairness_table(metric_template=[SQL TEMPLATE OF METRIC TABLE])
bf.generate_bias_fairness_metric(k=[K])
```
where 
- `WITH HISTORY?` specify whether the models are trained on the with or without history cohort
- `METRIC/FEATURE NAME` is the name of the features that you are interested in performing an audit (e.g., `income_category`)
- `REF GROUP` is the reference group in the comparison
- `DIRECTORY FOR MODELS` contains the model training output after running the pipeline
- `SQL TEMPLATE OF METRIC TABLE` is the template to pull metric data from the database (must contain a ID column and one feature column)
- `K` is the percent of population of interest


#### Q. What metrics are support by the `BiasFairness` evaluator?

Currently, we only support recall disparity ratio and false discovery rate ratio.


