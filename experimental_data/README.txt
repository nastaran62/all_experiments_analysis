1- prepare_data: read data from raw_data path,
   resample them based on input sampling rate,
   split based on marker and save in prepared_data path
   (for each experiment we save raw_data in a folder in this path and then save prepared data and preprocessed data in the same path)


2- preprocess_data:  read data from prepared_data and preprocess them and save in preprocessed_data
   preprocessing modules come from processing.preprocessing package

For each experiment we put one function in prepare_data and preprocess_data to consider differences


Before doing any feature extraction we should prepare data using this package
It may some projects needs more preparation, we should add extra preparation here specially for that projects
