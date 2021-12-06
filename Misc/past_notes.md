# EPA Team 1

[Project proposal notes](https://docs.google.com/document/d/1cRgkVr3T8WSK2C-aBW5zr6sd02lH8VDwu2UD7s6dZrA/edit)

[Project proposal document](https://www.overleaf.com/project/5f6bd3fd8984cd0001f7dd0d)

[Pipeline notes](https://docs.google.com/document/d/1uKHrK9bakVFbLholiuxpO89f4tOaLoM3lbSGly2_crw/edit)

[Pipeline slides](https://docs.google.com/presentation/d/1M4PYu4NoNAv8dtuoteM7gPedO_Pp5O_gsxorhvKzdhk/edit#slide=id.g9d94650c50_0_16)

[Modeling approach slides](https://docs.google.com/presentation/d/1zZOae65rr_Tb9E4hFDYto0p-EbpVj1UvS6KO05YWMNQ/edit#slide=id.g9d159e8a46_0_0)

[Feature list](https://docs.google.com/spreadsheets/d/1c9mi3LSiiS5z2OQWlm1ZIWf7zpTN4mbedvUyG7ccwPg/edit#gid=0)

[Validation splits and V0 results slides](https://docs.google.com/presentation/d/1rW_1OUCzOoqS5HxiD1DX7dvJW1LVTW2mhav_uILzzhk/edit#slide=id.g9f7e2762e5_0_0)

[Midterm project progress slides](https://docs.google.com/presentation/d/1Mvu-UhpycJesINh7usu2galS5sH_9gyKWPqy6csu4LY/edit#slide=id.g9ea6d034fe_0_0)

[Midterm project progress recording](https://drive.google.com/file/d/15LjixIpkWucZkv_KSIqO2_imgFpHNKqh/view)

[Pipeline updates slides](https://docs.google.com/presentation/d/1pm_l0RbRnY983UKjrt53c5pmQRxCw_Cnnacwownbt50/edit#slide=id.ga64c142357_0_6)

[Model grid and results slides](https://docs.google.com/presentation/d/1aiLNAzo9MQKEQd_LFACpUj2hvBaPI8rUgQWDbUYtaeU/edit#slide=id.g9fc8c439eb_0_0)

[Model selection and analysis slides](https://docs.google.com/presentation/d/1lrwuyMxjlTtlZokDbFIKLjrbGhopmgwb9iXfq8WrFwU/edit#slide=id.g541820e087_0_0) 

[Bias and fairness update slides](https://docs.google.com/presentation/d/1O-w_2-Ce7_xPUjGvkXHFyNvx2J4JWwXIgBGgpq9hkmA/edit#slide=id.ga2ce228026_0_0)

## Key Project Decisions Made

Decision: We will fit two models - one for handlers with previous inspection history and one for all handlers with previous inspection history features suppressed     
WHY? Requires less imputation, helps to address the sparsity issue, potentially more accurate  

Decision: Focus on LQGs rather than all types of handlers  
WHY? More reliable data on "active" status (annually, rather than biannualy)   

Decision: Binary classification rather than regression  
WHY? Likely more accurate, ability to refine labels allows for narrower/more actionable predictions

Decision: Use formal enforcements to narrow our label
WHY? Most inspections result in a violation, so we needed to narrow, see EDA

## Key Project Decisions To Be Made

What metrics are we using in addition to model validation metrics? How are we dealing with competing metrics?  
What baseline are we measuring against?  
When should the prediction be made each year? (currently January 1st)  
How are we testing the assumptions inherent in our modeling approach?  
What are the ethical implications of our label/feature selection choices?  
Are we thinking about equity from the perspective of facility owners or the community?  

## Key Assumptions

With regard to baseline and common sense baseline: We assume that the distribution of violations for inspected LQGs is representative of all LQGs  

## TODO

By Sunday
- Wenyu: change validation cohort to include all active handlers, implement support metric
- Wenyu: rerun baseline (with history and without)
- Carly: create plot of # of labeled examples in each validation cohort, look at cohort sizes
- Mike: update model grid (random forest hyperparameters, add primary_naics features, output feature importance variables for each model, remove logistic regression?)
- Mike: create procedure for subselecting models for second run (simpler, high precision, support: 50-75%) (after we review results from first run)
- Carly: create precision v. support plot w/two way error bars, feature importance bar plot

By Monday evening

By Tuesday morning


## COMPLETED

By Sunday night
- Wenyu traceback on one hot issue
- Carly to add new cohort template (all active handlers regardless of label, validation_cohort_template)
- Wenyu change the validation set to include new validation cohort with everyone
- Mike change output

By Monday morning
- Decide on hyperparameters (all - Mike will create a google sheet, Wenyu and Carly to review Monday morning)

By Tuesday noon
- Run grid and collect results

By Tuesday midnight
- Carly to create plot
- Update slides and submit (all)

By Tuesday
- Add transform (one hot) function (Carly)
- Add features

By Friday
- Add features(violation history - Wenyu) (naics - Carly to update) (industry info - Carly) (parent company - Carly)
- Add model types (decision trees - Mike) (random forest - Mike)
- Update baseline (number of violations - Wenyu) 
    * Need help! (I did pull out handler id, ranks by violation_number, but since evaluator changed (need logger, model name, save path etc.) having difficulty using evaluator to do prk graph)
- Update feature handling (#25 - Mike)

By Sunday
- Add precision plot, after we talk to Kit about what it actually is (??)
- Pick best models, update slides

Finish by Sunday evening (meet at 7:30pm):  
- test and debug scripts, run and collect V0 results (Wenyu)     
- update validation class and/or manually generate validation plots (Mike)    
- pull together draft midterm slides (Carly)

To prep for midterm presentation recording (record Sunday/submit Monday):   
- clean-up EDA to support project decisions (Mike - feature window, Carly - cohort, ? - percentage of handlers with inspection history per year)
- record presentation (all)

To turn in validation splits/V0 slides (Monday):  
- fill out slide template (mostly DONE)  

To turn in revisions from last week (Tuesday):
- sync new features list with old, make other updates as needed (all)   

Finish by Sunday morning (meet at 10am):
- ~~update feature templates to take cohort table name as parameter, left join on handler_id, get all feature templates (Wenyu)~~
- ~~update feature function (docstring) to format with table name (Mike)~~
- ~~update cohort template to create a table for each cohort (Mike)~~
- ~~update cohort function to return table name (Mike)~~
- ~~change base feature template to inspection history feature template (Mike)~~ 
- ~~add waste features that are already complete (Mike)~~
- ~~reformat feature templates to take one year date as a parameter (Mike)~~
- ~~update the config & execute script to handle features, update the way we specify train/val splits (Carly, Mike)~~
- ~~set params for V0 runs (Carly)~~  

Finish by Oct.23, Friday night:
- ~~recalculate baseline (prior) (Wenyu)~~ 
- ~~implement commonsense baseline (using number of days since last inspection, assign large value for those never inspected) (in one function: write SQL template (add to config), take rank, return handler ids and column of labels) (Wenyu)~~
- ~~finish feature templates for initial feature categories (Carly - industry info, Wenyu - violation history)~~  
- ~~write imputation functions as needed (Carly - create file in data folder, helper for flag (if n/a, 1; if not, 0)) (takes feature_df and list of features)~~ 
- ~~decide on validation splits for initial model instantiations (all) (DONE)~~
- ~~implement pr-k as validation function (Mike, Carly) (DONE)~~

To turn in project updates slides:
- create config (.yaml)
- update .gitignore to ignore only secrets.yaml
- move sql templates out of functions and into config
- move database utilities back out of dataprep
- update docstrings for dataprep functions
- create execute.py file to:
    - parse config
    - call dataprep functions (passing parameters from config)
    - instantiate model (passing parameters from config)
    - validate model & print or save results
- create pipeline diagram for slides

Pipeline functions:

*data prep*:
- Timesplitter (decide on how we will do validation splits after lecture tomorrow) (Carly)
    - Input: start time, end time, update time, prediction time
    - Output: pairs of <train start time, train end time, test start time, test end time>

- CohortCreator / ActiveHandlers (Wenyu)
     - Input: timesplitter output, cohort definition,[entity_ids, as of date]
     - Output: cohort matrix <entity_id, as_of_date>

- LabelCreator (Carly)
    - Input: pairs <entity_id, as_of_date>, label definition
    - Output: matrix <entity_id, as_of_date, label>

- FeatureCreator (several, split by schema.table?) (Carly)
    - Input: pairs <entity_id, as_of_date>, feature definition(s)
    - Output: matrix <entity_id, as_of_date, feature(s)>

- BuildMatrix (calls CohortCreator, LabelCreator, FeatureCreator)
    - Input: all the things
    - Output: full matrix

- TestTrainSplit (Carly)
    - Input: time pairs from TimeSplitter
    - Output: test matrix, train matrix
    - *Note from Mike: for now I require the data matrix to be a pandas dataframe with a column called 'label'*

*model classes*
- ModelTrainer (Mike √)
    - Input: test matrix, train matrix
    - Output: model object (stored), model definition

- Scorer (Mike √)
    - Input: model object, matrix, feature columns
    - prediction scores

*validation class*
- Evaluator (Mike)
    - Prediction scores, label column, metric(s)

