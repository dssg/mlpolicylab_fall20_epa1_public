# EPA Team 1

## Mitigating the Risk of Hazardous Waste Incidents with Machine Learning

#### Project Description

The New York State Department of Environmental Conservation (NYSDEC) is the sole authority responsible for the important task of enforcing hazardous waste regulations in New York State. Given the severe consequences that an hazardous waste incident can have for the health of the environment and surrounding community, proper oversight of New York's thousands of registered handlers is critically important. NYSDEC's primary mechanism for mitigating the risk of incidents is conducting handler inspections, through which violations are identified and corrected. However, with very limited resources, NYSDEC is only able to inspect a small fraction of registered handlers each year. Absent an oracle, it is difficult for inspection authorities to identify which handlers are at greatest risk of a causing a dangerous incident to prioritize them for inspection.

To support the goal of minimizing the number of hazardous waste incidents in New York State each year, we have developed a machine learning system that informs and enhances NYSDEC's inspection prioritization process. At the beginning of each year, our system selects the 300 active large quantity generator handlers most likely to be the subject of formal enforcement in that year and recommends them for inspection. To form this prioritized list, we combine the predictions of two separately trained models to select a total of 300 handlers from the exclusive groups of large quantity generators who have a history of inspections and those who do not, proportional to each group's size. This innovative approach helps address the selective labels issue, and allows NYSDEC to control the amount of resources they spend conducting repeat inspections of violators versus inspecting handlers who haven't yet incurred a violation, but are likely to.

Our top-performing models were evaluated based on their precision, recall, and "support", a metric representing the prevalence of large quantity generators for which we have previous inspection information in the model's selected group. Our results demonstrate that even at a relatively high threshold of support, our best models trained \textit{with} inspection history and \textit{without} inspection history are more efficient at identifying high-risk handlers than the commonsense baseline, which selects handlers based on their number of previous violations. In addition to evaluating these models for efficiency, we also provide NYSDEC with a secondary model evaluation method focused on fair allocation of the assistive benefits of inspections between low and high income communities.

#### File Structures

```
/
├── EDA/                                #scripts for EDA 
│   └── eda.sql                         
├── Misc/                               #miscellaneous things
│   ├── EPA1 FINAL.pdf                  
│   ├── epa1_project_proposal.pdf       
│   └── past_notes.md                  
├── Pipeline/                           #main Pipeline
│   ├── data/                           #data handling and processing
│   │   ├── acs_dataPrep.py
│   │   ├── data_notes.md
│   │   ├── dataPrep.py
│   │   ├── dataTransform.py
│   │   └── templates.yaml
│   ├── execute.py                      #pipeline executor
│   ├── experiment_config/              #experiment configurations
│   │   ├── dataConfig.yaml
│   │   ├── defaultConfig.yaml
│   │   ├── sampleConfig.yaml
│   ├── model/                          #model class
│   │   ├── baseModel.py
│   │   ├── decisionTreeModel.py
│   │   ├── logisticRegressionModel.py
│   │   ├── randomForestModel.py
│   │   ├── sklearnModel.py
│   │   ├── xgbClassifier.py
│   │   └── xgboostModel.py
│   ├── utils/                          #utility functions/classes
│   │   ├── databaseUtils.py
│   │   ├── loggingUtils.py
│   │   └── modelUtils.py
│   └── validation/                     #validation and model selection + bias/fairness audit
│       ├── common.ipynb
│       ├── eval.py
│       ├── modelSelect.py
│       └── readme.md
```

Please refer to `Pipeline` directory for more information on how to run the model grid, validate the models, and perform a bias and fairness audit.
