
# Shared parameters
phi = 0.002
cross_validation_folds = 20
random_seed = 20


# OLVF Parameters
olvf_C = 0.1
olvf_Lambda = 30
olvf_B = 0.64
olvf_option = 1 # 0, 1 or 2
stream_mode = "variable" # or variable, decrease sparsity when variable


# OLSF Parameters
olsf_C = 0.1
olsf_Lambda = 30
olsf_B = 0.64
olsf_option = 1 # 0, 1 or 2


# save parameter setting to a txt file
def saveParameters(figurename):
    text_file = open(figurename[:-3]+".txt", "w")
    param_setting = "SHARED - folds = "+str(cross_validation_folds) + "\n" + \
                    "SHARED - seed = "+str(random_seed) + "\n\n" + \
                    "OLVF - phi = "+str(phi) + "\n" + \
                    "OLVF - C = "+str(olvf_C) + "\n" + \
                    "OLVF - B = "+str(olvf_B) + "\n" + \
                    "OLVF - Lambda = "+str(olvf_Lambda) + "\n" + \
                    "OLVF - Option = "+str(olvf_option) + "\n" + \
                    "OLVF - StreamMode = "+stream_mode + "\n\n" + \
                    "OLSF - C = "+str(olsf_C) + "\n" + \
                    "OLSF - B = "+str(olsf_B) + "\n" + \
                    "OLSF - Lambda = "+str(olsf_Lambda) + "\n" + \
                    "OLSF - Option = "+str(olsf_option) + "\n"
    text_file.write(param_setting)
    text_file.close()



# 1 - Parameter Search
# 2 - Decaying Variance
# 3 - Sparsity, slack variable - do not add too many methods similar to OLSF if not necessary
# 4 - Maybe runtime comparison
# 5 - Magic variable features, float division by 0, debug



