import pandas as pd
import numpy as np
import preprocessing as preprocessing

tv = pd.read_csv("TVdata.txt")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

print(tv.head())
print(tv.info())

print(tv.drop([
    'video_id', 'import_id', 'release_year'], axis=1).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.95]))
print((tv == 0).sum(axis=0) / tv.shape[0])

tv.drop("video_id", inplace=True, axis=1)
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(tv['cvt_per_day'], bins=range(0, 15000, 30), color='r', label='cvt_per_day', density=True, alpha=0.5)
# 0 到15000的数据，30的区间，红色线，alpha是透明度
plt.legend(loc='upper right')
plt.title('Histograms of cvt_per_day before data processing')
plt.xlabel('cvt_per_day')
plt.ylabel('density')
plt.show()

corr = tv.corr()
sns.heatmap(corr, cmap='YlGnBu')
plt.show()

plt.ylim(15000)
sns.boxplot(x='import_id', y='cvt_per_day', data=tv)
plt.show()

plt.ylim(15000)
sns.boxplot(x='mpaa', y='cvt_per_day', data=tv)
plt.show()

plt.ylim(15000)
sns.boxplot(x='awards', y='cvt_per_day', data=tv)
plt.show()

gen_split = tv['genres'].str.get_dummies(sep=',').sum()
gen_split.sort_values(ascending=False).plot.bar()
plt.show()

plt.hist(tv['release_year'].values, bins=range(1917, 2017, 10), alpha=0.5, color='r', label='release_year')
plt.legend(loc='upper left')
plt.title("release year before data processing")
plt.xlabel('release_year')
plt.ylabel('Count')
plt.show()

# convert categorical variables into dummy variables (one hot encoding)
d_import_id = pd.get_dummies(tv['import_id']).astype(np.int64)
d_mpaa = pd.get_dummies(tv['mpaa']).astype(np.int64)
d_awards = pd.get_dummies(tv['awards']).astype(np.int64)

# convert generas into dummy
d_genes = tv['genres'].str.get_dummies(sep=",").astype(np.int64)
d_genes["other"] = d_genes['Anime'] | d_genes['Reality'] | d_genes['Lifestyle'] | d_genes['Adult'] | d_genes['LGBT'] | \
                   d_genes["Holiday"]
d_genes.drop(['Anime', 'Reality', 'Lifestyle', 'Adult', 'LGBT', "Holiday"], inplace=True, axis=1)
print(tv['release_year'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

bin_year = [1916, 1974, 1991, 2001, 2006, 2008, 2010, 2012, 2013, 2014, 2017]
year_range = ['1991-1974', '1974-1991', '1991-2001', '2001-2006', '2006-2008', '2008-2010', '2010-2012', '2012-2013',
              '2013-2014', '2014-2017']
year_bin = pd.cut(tv['release_year'], bin_year, labels=year_range)
d_year = pd.get_dummies(year_bin).astype(np.int64)
tv.drop(['import_id', 'mpaa', 'awards', 'genres', 'release_year'], axis=1, inplace=True)
tv = pd.concat([tv, d_genes, d_mpaa, d_year, d_awards, d_import_id], axis=1)
print(tv.head())

tv[['budget', 'boxoffice', 'metacritic_score', 'star_category', 'imdb_votes', 'imdb_rating']] = tv[
    ['budget', 'boxoffice', 'metacritic_score', 'star_category', 'imdb_votes', 'imdb_rating']].replace(0, np.nan)

tv['budget'] = tv['budget'].fillna(tv['budget'].mean())
tv['boxoffice'] = tv['boxoffice'].fillna(tv['boxoffice'].mean())
tv['metacritic_score'] = tv['metacritic_score'].fillna(tv['metacritic_score'].mean())
tv['star_category'] = tv['star_category'].fillna(tv['star_category'].mean())
tv['imdb_votes'] = tv['imdb_votes'].fillna(tv['imdb_votes'].mean())
tv['imdb_rating'] = tv['imdb_rating'].fillna(tv['imdb_rating'].mean())
print(tv.info())

# standard scaling
scale_lst = ['weighted_categorical_position', 'weighted_horizontal_poition', 'budget', 'boxoffice',
             'imdb_votes', 'imdb_rating', 'duration_in_mins', 'metacritic_score', 'star_category']
newTv = tv.copy()
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing

# standarlization
scale = StandardScaler()
sc_scale = scale.fit(newTv[scale_lst])
newTv[scale_lst] = sc_scale.transform(newTv[scale_lst])

# minmax scale

newTv2 = tv.copy()
mm_scale = preprocessing.MinMaxScaler().fit(newTv2[scale_lst])
newTv2[scale_lst] = mm_scale.transform(newTv2[scale_lst])

# Robust scaling

newTv3 = tv.copy()
rb_scale = preprocessing.RobustScaler().fit(newTv3[scale_lst])
newTv3[scale_lst] = rb_scale.transform(newTv3[scale_lst])

# model training

from sklearn import model_selection

X = newTv.drop(['cvt_per_day'], axis=1)
Y = newTv['cvt_per_day']
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=5)

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

classifierLasso = Lasso()
classifierRidge = Ridge()
classifierRandomForest = RandomForestRegressor()

models_names = ["classifierLasso", "classifierRandomForest", "classifierRidge"]
model_list = [classifierLasso, classifierRandomForest, classifierRidge]
count = 0


for classifier in model_list:
    cv_score = model_selection.cross_val_score(classifier, x_train, y_train, cv=10)
    print(cv_score)
    print('Model Accuracy of ' + models_names[count] + ' is ' + str(cv_score.mean()))
    count += 1

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results
def print_grid_search_metrics(gs):
    print ("Best score: " + str(gs.best_score_))
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))

parameters={
    'alpha':[61,62,62.1,62.2,62.5]
}

Grid_Lasso = GridSearchCV(Lasso(),parameters, cv = 5)
Grid_Lasso.fit(x_train,y_train)
print_grid_search_metrics(Grid_Lasso)

parameters={
'alpha':[220,221,222,223,224,224.1,224.2,224.3,224.4,224.5,225,226]
}
Grid_Ridge = GridSearchCV(Ridge(),parameters, cv = 5)
Grid_Ridge.fit(x_train,y_train)
print_grid_search_metrics(Grid_Ridge)
