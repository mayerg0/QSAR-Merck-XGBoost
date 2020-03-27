# QSAR-Merck-XGBoost

## Description
Repo comparing the performance of DNN to XGBoost for QSAR applications using Merck 2013 dataset

## Installation
- Install requirements
- Download data from [Merck Dataset](https://www.kaggle.com/c/MerckActivity/data "Merck Dataset") into Data folder
- run "python main.py arg_dataset arg_model". arg_dataset can be "all" or an int between 1 and 15, arg_model can be "dnn", "rf" or "xgb"

## Resources
- Merck Paper:
Ma, J., Sheridan, R.P., Liaw, A., Dahl, G.E. and Svetnik, V., 2015. Deep neural nets as a method for quantitative structure–activity relationships. Journal of chemical information and modeling, 55(2), pp.263-274.

- DNN inspired from:
[DNN repo MIT project](https://github.com/RuwanT/merck/blob/master/custom_networks.py)

- Project Presentation:
[Molecular Activity Prediction with XGBoost](https://docs.google.com/presentation/d/e/2PACX-1vRN4hVoXzNEVSYAOq9eYWDQxudlwCAL2GNW9Mx1D7ScT6pOXDTEmxUOeV_jnOo__zi9hKX_yoLPq0R6/pub?start=false&loop=false&delayms=3000)
