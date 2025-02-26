# Transport-based transfer learning on Electronic Health Records throught optimal transport (OTTEHR): Application to detection of treatment disparities #

Datasets for all experiments including MIMIC-III, MIMIC-IV and eICU are obtained under the license of physionet. 

## Insurance and Age experiments 

Run exp/mimic_iii/admission/run_OTTEHR.py to run OTTEHR on insurance experiments.

Run exp/mimic_iii/admission/run_OTTEHR_with_age.py to run OTTEHR on age experiments. 

Run exp/mimic_iii/admission/analyze_bound.ipynb to generate analysis of target error and individual terms in the derived upper bound for insurance experiments.

Run exp/mimic_iii/admission/analyze_accuracy.ipynb to generate benchmarking results of OTTEHR against existing transfer learning methods for insurance and age experiments with **group_name** and **groups** updated to appropriate values. 

Run exp/mimic_iii/admission/analyze_duration_diff.ipynb to generate predicted duration vs observed duration and the treatment disparity analysis based on subgroups for insurance experiments.

Run exp/mimic_iii/admission/analyze_dist_shift.ipynb to analyze distributional shifts in insurance and age experiments with **group_name** and **groups** updated to appropriate values. 


## Cross-database experiments

Run exp/cross/run_OTTEHR.py to run OTTEHR on cross-database experiments.

Run exp/cross/analyze_accuracy.ipynb to generate benchmarking results of OTTEHR against existing transfer learning methods. 

Run exp/cross/analyze_dist_shift.ipynb to analyze distributional shifts in cross-database experiments.

## Cross-hospital experiments

Run exp/eICU/run_OTTEHR.py to run OTTEHR on cross-database experiments.

Run exp/eICU/analyze_accuracy.ipynb to generate benchmarking results of OTTEHR against existing transfer learning methods. 

Run exp/eICU/analyze_dist_shift.ipynb to analyze distributional shifts in cross-database experiments.

