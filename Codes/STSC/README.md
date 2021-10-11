To run the STSC code:
* Open the GetConfiguration file and select the TargetLabeledRatio based on the number of days that you would like to test.
* run the following instruction: config = GetConfiguration('TSC+SVM')
* run the following instruction: CrossValidation(config, 10,8) % where cv = 10 and seed = 8
