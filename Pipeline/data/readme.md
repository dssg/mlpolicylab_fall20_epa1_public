## Data

#### File Structure
```
├── data/
│   ├── acs_dataPrep.py
│   ├── dataPrep.py
│   ├── dataTransform.py
│   └── templates.yaml
```

##### `acs_dataPrep.py` 
This script can automatically pull variables specified from the American Community Survey (ACS) on a block level.
After selecting variables that you are interested in, simply run 

```
$ python acs_dataPrep.py
```

the file with the following changes in the main function:

```
variables = [VAR1, VAR2, VAR3]

block_iter_query_write(OUTPUT PATH,
                       variables=variables,
                       state=state2index[STATE OF INTEREST])
```
and the data will be available in `.csv` form under the path you specified.


##### `dataPrep.py` and `dataTransform.py`

_Note: These files are not meant to be run separately._

This two files contains the code to automatically construct the training/validation cohort, labels, and features by communicating with `PostgreSQL` database.
It can also perform a series of data pre-process procedures such as one-hot encoding, feature normalization, missing value imputation, and etc. 

##### `tempates.yaml` 

This file contains the `SQL` templates for our cohort, labels, and features.