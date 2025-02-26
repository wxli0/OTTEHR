from ast import literal_eval
import pandas as pd

import getpass
user_id = getpass.getuser()

def construct_freq_dict_group(df, dividing_feature, group_1, group_2):
    """ 
    Construct frequency dictionary dictionary for group 1 and group 2,
    considering the duplicate codes for one admission as one code
    """

    group_1_df = df[df[dividing_feature] == group_1]
    group_2_df = df[df[dividing_feature] == group_2]

    def construct_freq_dict(group_df):
        group_dict = {}
        for _, row in group_df.iterrows():
            code_set = list(set(row['ICD codes']))
            for code in code_set:
                if code not in group_dict:
                    group_dict[code] = 1
                else:
                    group_dict[code] += 1
        return group_dict
    
    group_1_dict = construct_freq_dict(group_1_df)
    group_2_dict = construct_freq_dict(group_2_df)

    # add code with zero counts to each dictionary
    for code in group_1_dict.keys():
        if code not in group_2_dict:
            group_2_dict[code] = 0
    
    for code in group_2_dict.keys():
        if code not in group_1_dict:
            group_1_dict[code] = 0
    
    return group_1_dict, group_2_dict


def select_codes(group_1_dict, group_2_dict, group_1_min_count, group_2_min_count):
    """ 
    Select codes for group 1 and group 2 using their minimum count
    """

    all_codes = list(group_1_dict.keys())
    group_2_codes = list(group_2_dict.keys())
    all_codes.extend(group_2_codes)
    all_codes = list(set(all_codes))

    def filter_codes(group_1_dict, group_2_dict, group_1_min_count, group_2_min_count):
        """ 
        Codes filtered for group 1 and group 2 using their minimum count
        """
        def filtered_code_group(group_dict, min_count):
            """ 
            Codes filtered out for a group since the number is too few
            """
            filtered_codes = []
            for code, value in group_dict.items():
                if value < min_count:
                    filtered_codes.append(code)
            return filtered_codes
        filtered_group_1_codes = filtered_code_group(group_1_dict, group_1_min_count)
        filtered_group_2_codes = filtered_code_group(group_2_dict, group_2_min_count)
        filtered_codes = filtered_group_1_codes
        filtered_codes.extend(filtered_group_2_codes)
        return list(set(filtered_codes))

    filtered_codes = filter_codes(group_1_dict, group_2_dict, group_1_min_count, group_2_min_count)
    selected_codes = [x for x in all_codes if x not in filtered_codes]
    return selected_codes



df_path = f"/home/{user_id}/OTTEHR/outputs/mimic/admission_patient_diagnosis_ICD.csv"
admid_diag_df = pd.read_csv(df_path, index_col=0, header=0, converters={'ICD codes': literal_eval})
admid_diag_df

male_freq_dict, female_freq_dict = construct_freq_dict_group(admid_diag_df, 'gender', 'M', 'F')
male_min_count = 120
female_min_count = 100
selected_codes = select_codes(male_freq_dict, female_freq_dict, male_min_count*2, female_min_count*2)



