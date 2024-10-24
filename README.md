# bp

This is a repository for bachelor's thesis: Utilization of Transformer architecture for predicting financial time series in the forex market


This project contains two models implemented from scratch: the Transformer and Autoformer.

Directories:
data -- contains three CSV datasets that served as input data for the models.
saved_models -- several models created, trained, and saved.
src -- source code for the Transformer and Autoformer model as well as classes used for loading the datasets, pre-processing data, evaluating models,  training the models, and visualizations.

Ultimately, the models that are presented here, were trained to predict two different target variables:
1. a sequence of prices of a forex currency pair
2. a sequence of changes in a pair's price

Notebooks involved in price prediction are:
1. autoformer_predict_price.ipynb
2. transformer_predict_price.ipynb

Notebooks involved in price change prediction are:
1. autoformer_predict_price_move.ipynb
2. transformer_in_seq.ipynb
3. transformer_out_seq.ipynb