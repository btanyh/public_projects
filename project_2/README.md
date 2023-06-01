# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Singapore Housing Data and Kaggle Challenge
## Overview
The goal of this project is to provide first time home buyers in Singapore a good gauge on the price they should be paying for their flats without having external features affecting the prices.

This will be accomplished trough the use of Machine Learning models, specifically Linear, Lasso, Ridge, and ElasticNet Regression.

Success of the models will be evaluated through R-Squared, and RMSE scores.

The project will go through the full modelling workflow, conclusions, and recommendations for future projects.

---

### Problem Statement

In Singapore, a majority [77.9%](https://www.singstat.gov.sg/find-data/search-by-theme/households/households/latest-data) of the population live in public housing flats. These are priced according to HDB themselves [*by establishing the market value of the flat by looking at the prices of comparable resale flats nearby, which is influenced by **prevailing market conditions**, as well as the individual attributes of the flats.*](https://www.hdb.gov.sg/cs/infoweb/about-us/news-and-publications/publications/hdbspeaks/How-BTO-Flats-are-Priced#:~:text=First%2C%20HDB%20establishes%20the%20market,individual%20attributes%20of%20the%20flats.&text=Projects%20with%20locational%20advantages%20will,to%20city%20or%20town%20centre)

However, this way of pricing can leave first time home buyers paying amounts which are far above what the flat is actually worth due to 'prevailing market conditions'. Examples of such market conditions would be [pent-up demand due to COVID](https://www.businesstimes.com.sg/companies-markets/energy-commodities/prices-core-construction-materials-singapore-remain-high-next), or an [increase in prices of construction materials](https://www.businesstimes.com.sg/companies-markets/energy-commodities/prices-core-construction-materials-singapore-remain-high-next). 

I aim to offer first time home buyers a way of getting a good gauge of property prices based on the attributes of the flat and property location, and not exacerbated costs due to external factors.

---

### Datasets

There are 2 datasets included in the `data/inputs` folder. Details shown below.

* `train.csv`: this data contains all of the training data for the model which has over 70 features of Singapore Housing.
* `test.csv`: this data contains the test data for the model

---

### Data Dictionary

|Feature|Type|Description|
|:---|:---|:---| 
|resale_price|float|the property's sale price in Singapore dollars. This is the target variable that you're trying to predict for this challenge.|
|Tranc_YearMonth|string|year and month of the resale transaction, e.g. 2015-02|
|town|string|HDB township where the flat is located, e.g. BUKIT MERAH|
|flat_type|string|type of the resale flat unit, e.g. 3 ROOM|
|block|string|block number of the resale flat, e.g. 454|
|street_name|string|street name where the resale flat resides, e.g. TAMPINES ST 42|
|storey_range|string|floor level (range) of the resale flat unit, e.g. 07 TO 09|
|floor_area_sqm|float|floor area of the resale flat unit in square metres|
|flat_model|string|HDB model of the resale flat, e.g. Multi Generation|
|lease_commence_date|integer|commencement year of the flat unit's 99-year lease|
|Tranc_Year|integer|year of resale transaction|
|Tranc_Month|integer|month of resale transaction|
|mid_storey|integer|median value of storey_range|
|lower|integer|lower value of storey_range|
|upper|integer|upper value of storey_range|
|mid|integer|middle value of storey_range|
|full_flat_type|string|combination of flat_type and flat_model|
|address|string|combination of block and street_name|
|floor_area_sqft|float|floor area of the resale flat unit in square feet|
|hdb_age|integer|number of years from lease_commence_date to present year|
|max_floor_lvl|integer|highest floor of the resale flat|
|year_completed|integer|year which construction was completed for resale flat|
|residential|boolean|boolean value if resale flat has residential units in the same block|
|commercial|boolean|boolean value if resale flat has commercial units in the same block|
|market_hawker|boolean|boolean value if resale flat has a market or hawker centre in the same block|
|multistorey_carpark|boolean|boolean value if resale flat has a multistorey carpark in the same block|
|precinct_pavilion|boolean|boolean value if resale flat has a pavilion in the same block|
|total_dwelling_units|integer|total number of residential dwelling units in the resale flat|
|1room_sold|integer|number of 1-room residential units in the resale flat|
|2room_sold|integer|number of 2-room residential units in the resale flat|
|3room_sold|integer|number of 3-room residential units in the resale flat|
|4room_sold|integer|number of 4-room residential units in the resale flat|
|5room_sold|integer|number of 5-room residential units in the resale flat|
|exec_sold|integer|number of executive type residential units in the resale flat block|
|multigen_sold|integer|number of multi-generational type residential units in the resale flat block|
|studio_apartment_sold|integer|number of studio apartment type residential units in the resale flat block|
|1room_rental|integer|number of 1-room rental residential units in the resale flat block|
|2room_rental|integer|number of 2-room rental residential units in the resale flat block|
|3room_rental|integer|number of 3-room rental residential units in the resale flat block|
|other_room_rental|integer|number of "other" type rental residential units in the resale flat block|
|postal|integer|postal code of the resale flat block|
|Latitude|float|Latitude based on postal code|
|Longitude|float|Longitude based on postal code|
|planning_area|string|Government planning area that the flat is located|
|Mall_Nearest_Distance|float|distance (in metres) to the nearest mall|
|Mall_Within_500m|float|number of malls within 500 metres|
|Mall_Within_1km|float|number of malls within 1 kilometre|
|Mall_Within_2km|float|number of malls within 2 kilometres|
|Hawker_Nearest_Distance|float|distance (in metres) to the nearest hawker centre|
|Hawker_Within_500m|float|number of hawker centres within 500 metres|
|Hawker_Within_1km|float|number of hawker centres within 1 kilometre|
|Hawker_Within_2km|float|number of hawker centres within 2 kilometres|
|hawker_food_stalls|integer|number of hawker food stalls in the nearest hawker centre|
|hawker_market_stalls|integer|number of hawker and market stalls in the nearest hawker centre|
|mrt_nearest_distance|float|distance (in metres) to the nearest MRT station|
|mrt_name|string|name of the nearest MRT station|
|bus_interchange|boolean|boolean value if the nearest MRT station is also a bus interchange|
|mrt_interchange|boolean|boolean value if the nearest MRT station is a train interchange station|
|mrt_latitude|float|latitude (in decimal degrees) of the the nearest MRT station|
|mrt_longitude|float|longitude (in decimal degrees) of the nearest MRT station|
|bus_stop_nearest_distance|float|distance (in metres) to the nearest bus stop|
|bus_stop_name|string|name of the nearest bus stop|
|bus_stop_latitude|float|latitude (in decimal degrees) of the the nearest bus stop|
|bus_stop_longitude|float|longitude (in decimal degrees) of the nearest bus stop|
|pri_sch_nearest_distance|float|distance (in metres) to the nearest primary school|
|pri_sch_name|string|name of the nearest primary school|
|vacancy|integer|number of vacancies in the nearest primary school|
|pri_sch_affiliation|boolean|boolean value if the nearest primary school has a secondary - school affiliation|
|pri_sch_latitude|float|latitude (in decimal degrees) of the the nearest primary school|
|pri_sch_longitude|float|longitude (in decimal degrees) of the nearest primary school|
|sec_sch_nearest_dist|float|distance (in metres) to the nearest secondary school|
|sec_sch_name|string|name of the nearest secondary school|
|cutoff_point|integer|PSLE cutoff point of the nearest secondary school|
|affiliation|boolean|boolean value if the nearest secondary school has an primary school affiliation|
|sec_sch_latitude|float|latitude (in decimal degrees) of the the nearest secondary school|
|sec_sch_longitude|float|longitude (in decimal degrees) of the nearest secondary school|

---

### Analysis

The project has been broken down into 4 code notebooks as described below.

Under the `code` folder for this project.
1. `01_eda_and_cleaning`: The initial training dataset is loaded, and eda and data cleaning is done.
1. `02_preprocessing_and_initial_models`: Preprocessing such as scaling and encoding is done before fitting the data to the baseline model and regularisation regression models.
1. `03_model_tuning`: Using the results from the initial models, more feature engineering is done, and models are refitted to see if performance is improved.
1.  `04_kaggle_submission`: Final selection of model, training the model on the entire training set, submission of final model to Kaggle.


The results from the models are listed below.

The results are good, as they all have an R-Squared score of 0.89 - 0.90. The RMSE scores are also all within the 45000 range. This means the model's predicted price of properties are on average only 45000 dollars off the actual price. This result in terms of property prices, which can go up into the millions in my opinion is a good one.

The final model which was used for the Kaggle submission was the Ridge regression model without removing outliers as it had the best performing RMSE score, and it makes sense to use a model which has regularisation in this context with a dataset of over 70 features.

| Model                                | R-Squared Score - Training Set | R-Squared Score - Test Set | Cross-Validated R-Squared Score folds = 5 | RMSE  |
|--------------------------------------|--------------------------------|----------------------------|-------------------------------------------|-------|
| OLS                                  | 0.90                           | 0.90                       | 0.90                                      | 44827 |
| Lasso  alpha = 93.6                  | 0.89                           | 0.89                       | 0.89                                      | 47068 |
| Ridge alpha = 1                      | 0.90                           | 0.90                       | 0.90                                      | 44818 |
| ElasticNet l1_ratio = 1 alpha = 93.6 | 0.89                           | 0.89                       | 0.89                                      | 47068 |



| Model after removing outliers and 0 weighted features by Lasso| R-Squared Score - Training Set | R-Squared Score - Test Set | Cross-Validated R-Squared Score folds = 5 | RMSE  |
|--------------------------------------|--------------------------------|----------------------------|-------------------------------------------|-------|
| OLS                                  | 0.90                           | 0.90                       | 0.90                                      | 45379 |
| Lasso  alpha = 93.6                  | 0.89                           | 0.89                       | 0.89                                      | 46429 |
| Ridge alpha = 1                      | 0.90                           | 0.90                       | 0.90                                      | 45383 |
| ElasticNet l1_ratio = 1 alpha = 93.6 | 0.89                           | 0.89                       | 0.89                                      | 46429 |

---

### Conclusions & Recommendations

To conclude, in the context of the problem statement, this project has accomplished its goal of offering first time home buyers a way of getting a good gauge of property prices based on the attributes of the flat and property location, and not exacerbated costs due to external factors.

First time home buyers can now use this model to input features of BTO flats that they are interested in to get a good estimate of what the flat is worth regardless of the market conditions during their time.

Some recommendations for future projects to futher improve upon this model are listed below:
1. To include demographics of the populations of the respective areas. For example, to include the average age of the residents in a particular area. If an area has a large amount of retirees, there could be less demand for higher priced properties, which could lower the average price.
1. To include an average decibel level as a feature. An example would be if a property is directly located under a busy flight path, with constant noise pollution. This could affect the property price as well.
1. Proximity to parks / green spaces. Some properties which are closer to parks may command a premium in terms of price. This data should be added in future projects.

