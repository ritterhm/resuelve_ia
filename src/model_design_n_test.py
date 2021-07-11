#!/usr/bin env python

# ===============================================================================
# Model build and testing for design purposes.
#
# Author: Miguel Ángel de la Rosa García
#
# This file is part of Prueba IA Resuelve project.
# ===============================================================================


# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------

#Third party libraries
import numpy
import pandas
import sklearn.linear_model
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.preprocessing
import sklearn.tree

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LABEL_SIZE = 50
TESTS_ROUNDS = 20

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

# Load test data and print some dat to enable visual examination of data
data = pandas.read_csv('../data/datos_prestamo.csv')
print('Shape'.center(LABEL_SIZE, '-'))
print(data.shape)
print()
print('Info'.center(LABEL_SIZE, '-'))
print(data.info())
print()
print('Describe'.center(LABEL_SIZE, '-'))
print(data.describe())
print()
print('Nulls'.center(LABEL_SIZE, '-'))
print(data.isnull().sum())
print()
print('Sample data'.center(LABEL_SIZE, '-'))
print(data.head(50))

print()
print('Credit'.center(LABEL_SIZE, '-'))
print(pandas.crosstab(data['Historial_crediticio'], data['Estatus_prestamo']))
print()
print('-' * LABEL_SIZE)
print(pandas.crosstab(data['Salario'], data['Credito_pedido']))
print()
print('-' * LABEL_SIZE)
print(pandas.crosstab(data['Plazo_prestamo'], data['Credito_pedido']))
print()
print('-' * LABEL_SIZE)
print(pandas.crosstab(data['Casado'], data['Estatus_prestamo']))
print()
print('-' * LABEL_SIZE)
print(pandas.crosstab(data['Dependientes'], data['Estatus_prestamo']))
print()
print('-' * LABEL_SIZE)
print(pandas.crosstab(data['Educacion'], data['Estatus_prestamo']))

print()
print('Education'.center(LABEL_SIZE, '-'))
print(pandas.crosstab(data['Salario'], data['Educacion']))

# Filling null data
data['Genero'].fillna(data['Genero'].mode()[0], inplace=True)
data['Casado'].fillna(data['Casado'].mode()[0], inplace=True)
data['Dependientes'].fillna(data['Dependientes'].mode()[0], inplace=True)
data['Trabaja_para_el'].fillna(data['Trabaja_para_el'].mode()[0], inplace=True)
data['Credito_pedido'].fillna(data['Credito_pedido'].mean(), inplace=True)
data['Plazo_prestamo'].fillna(data['Plazo_prestamo'].mode()[0], inplace=True)
data['Historial_crediticio'].fillna(data['Historial_crediticio'].mode()[0], inplace=True)

print()
print('Nulls'.center(LABEL_SIZE, '-'))
print(data.isnull().sum())

#Adding total income column to facilitate classification
data['Ingreso_total'] = data['Salario'] + data['Salario_Pareja']

# Adding normalized values (log) to avoid jumps in standard deviation in wages and loans
data['Credito_pedido_log'] = numpy.log(data['Credito_pedido'])
data['Ingreso_total_log'] = numpy.log(data['Ingreso_total'])

# Show available columns
print(data.columns)

# Generating data model
model_variables = data[[
	'Genero', 'Casado', 'Dependientes', 'Educacion', 'Trabaja_para_el',
	'Plazo_prestamo', 'Historial_crediticio', 'Area_vivienda',
	'Credito_pedido_log', 'Ingreso_total_log'
]].values
model_results = data.iloc[:,15].values

# Label numeric transformation
print('Sample data before label numeric transform'.center(LABEL_SIZE, '-'))
print(model_variables)
print(model_results[:20])

encoder = sklearn.preprocessing.LabelEncoder()

for i in range(0, 8):
	if i == 5:
		# Plazo de préstamo
		continue

	model_variables[:,i] = encoder.fit_transform(model_variables[:,i])

model_results = encoder.fit_transform(model_results)

print('Sample data after label numeric transform'.center(LABEL_SIZE, '-'))
print(model_variables)
print(model_results[:20])

## Train model

#print()
#print('Train Model'.center(LABEL_SIZE, '-'))
#train_var, test_var, train_res, test_res = sklearn.model_selection.train_test_split(
	#model_variables, model_results, test_size=0.5
#)
#print(train_var)
#print(train_res)
#print(test_var)
#print(test_res)

## Testing accuracy with some algorithms

#scaler = sklearn.preprocessing.StandardScaler()
#train_var = scaler.fit_transform(train_var)
#test_var = scaler.fit_transform(test_var)

# Logistic Regression
precission = 0

for i in range(TESTS_ROUNDS):
	# Train data
	train_var, test_var, train_res, test_res = sklearn.model_selection.train_test_split(
		model_variables, model_results, test_size=0.5
	)
	scaler = sklearn.preprocessing.StandardScaler()
	train_var = scaler.fit_transform(train_var)
	test_var = scaler.fit_transform(test_var)

	# ML tester
	lregression = sklearn.linear_model.LogisticRegression()
	lregression.fit(train_var, train_res)
	prediction = lregression.predict(test_var)
	print('Linear regression accuracy {}: {}'.format(i, sklearn.metrics.accuracy_score(prediction, test_res)))
	precission += sklearn.metrics.accuracy_score(prediction, test_res)
else:
	print('Linear regression mean accuracy: {}'.format(precission / TESTS_ROUNDS))


# Decision Tree
precission = 0

for i in range(TESTS_ROUNDS): # Winner 86.63% percent accuracy with low variation across tests
	# Train data
	train_var, test_var, train_res, test_res = sklearn.model_selection.train_test_split(
		model_variables, model_results, test_size=0.5
	)
	scaler = sklearn.preprocessing.StandardScaler()
	train_var = scaler.fit_transform(train_var)
	test_var = scaler.fit_transform(test_var)

	# ML tester
	dtree = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
	dtree.fit(train_var, train_res)
	prediction = dtree.predict(test_var)
	print('Decision tree accuracy {}: {}'.format(i, sklearn.metrics.accuracy_score(prediction, test_res)))
	precission += sklearn.metrics.accuracy_score(prediction, test_res)
else:
	print('Decision tree mean accuracy: {}'.format(precission / TESTS_ROUNDS))

# Bayes Gaussian
precission = 0

for i in range(TESTS_ROUNDS):
	# Train data
	train_var, test_var, train_res, test_res = sklearn.model_selection.train_test_split(
		model_variables, model_results, test_size=0.5
	)
	scaler = sklearn.preprocessing.StandardScaler()
	train_var = scaler.fit_transform(train_var)
	test_var = scaler.fit_transform(test_var)

	# ML tester
	gaussian = sklearn.naive_bayes.GaussianNB()
	gaussian.fit(train_var, train_res)
	prediction = gaussian.predict(test_var)
	print('Bayesian gaussian accuracy {}: {}'.format(i, sklearn.metrics.accuracy_score(prediction, test_res)))
	precission += sklearn.metrics.accuracy_score(prediction, test_res)
else:
	print('Bayesian gaussian mean accuracy: {}'.format(precission / TESTS_ROUNDS))
