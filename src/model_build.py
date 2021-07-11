#!/usr/bin env python

# ===============================================================================
# Model builder.
#
# Author: Miguel Ángel de la Rosa García
#
# This file is part of Prueba IA Resuelve project.
# ===============================================================================


# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------

#Third party libraries
import joblib
import pandas
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing

# Project libraries
import common


# -----------------------------------------------------------------------------
# Model building
# -----------------------------------------------------------------------------

data = pandas.read_csv('../data/datos_prestamo.csv')
encoder = sklearn.preprocessing.LabelEncoder()

common.data_transform(data)
model_variables = common.model_input_get(data)
common.labels_encode(model_variables)
model_results = data.iloc[:,15].values
model_results = encoder.fit_transform(model_results)

# Train Model
train_var, test_var, train_res, test_res = sklearn.model_selection.train_test_split(
	model_variables, model_results, test_size=0.5
)
scaler = sklearn.preprocessing.StandardScaler()
train_var = scaler.fit_transform(train_var)
test_var = scaler.fit_transform(test_var)

model = sklearn.linear_model.LogisticRegression()
model.fit(train_var, train_res)

# Save model
joblib.dump(model, '../release/model.ml')

# Test model
model = joblib.load('../release/model.ml')
prediction = model.predict(test_var)
print(prediction)
print('Accuracy: ', sklearn.metrics.accuracy_score(prediction, test_res))
