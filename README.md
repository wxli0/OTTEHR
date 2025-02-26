# Transport-based transfer learning on Electronic Health Records throught optimal transport (OTTEHR): Application to detection of treatment disparities #

## Insurance and Age experiments 

Run mimic_iii_exp/admission/run_OTTEHR.py to run OTTEHR on insurance experiments.

Run mimic_iii_exp/admission/run_OTTEHR_with_age.py to run OTTEHR on age experiments. 

Run mimic_iii_exp/admission/analyze_bound.ipynb to generate analysis of target error and individual terms in the derived upper bound for insurance experiments.

Run mimic_iii_exp/admission/analyze_accuracy.ipynb to generate benchmarking results of OTTEHR against existing transfer learning methods for insurance and age experiments with **group_name** and **groups** updated to appropriate values. 

Run mimic_iii_exp/admission/analyze_duration_diff.ipynb to generate predicted duration vs observed duration and the treatment disparity analysis based on subgroups for insurance experiments.

Run mimic_iii_exp/admission/analyze_dist_shift.ipynb to analyze distributional shifts in insurance and age experiments with **group_name** and **groups** updated to appropriate values. 


## Cross-database experiments

Run cross_exp/run_OTTEHR.py to run OTTEHR on cross-database experiments.

Run cross_exp/analyze_accuracy.ipynb to generate benchmarking results of OTTEHR against existing transfer learning methods. 

Run cross_exp/analyze_dist_shift.ipynb to analyze distributional shifts in cross-database experiments.
