# ===============================================================================
# Common functions.
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
import sklearn.preprocessing


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def data_transform(data:object) -> None:
	"""
		Fills suceptible empty data and adds normalized information to input
		data in place.

		Arguments:
			object data:
				Pandas input dataframe to be transformed.

		Returns:
			None.
	"""

	# Filling null data
	data['Genero'].fillna(data['Genero'].mode()[0], inplace=True)
	data['Casado'].fillna(data['Casado'].mode()[0], inplace=True)
	data['Dependientes'].fillna(data['Dependientes'].mode()[0], inplace=True)
	data['Trabaja_para_el'].fillna(data['Trabaja_para_el'].mode()[0], inplace=True)
	data['Credito_pedido'].fillna(data['Credito_pedido'].mean(), inplace=True)
	data['Plazo_prestamo'].fillna(data['Plazo_prestamo'].mode()[0], inplace=True)
	data['Historial_crediticio'].fillna(data['Historial_crediticio'].mode()[0], inplace=True)

	#Adding total income column to facilitate classification
	data['Ingreso_total'] = data['Salario'] + data['Salario_Pareja']

	# Adding normalized values (log) to avoid jumps in standard deviation in wages and loans
	data['Credito_pedido_log'] = numpy.log(data['Credito_pedido'])
	data['Ingreso_total_log'] = numpy.log(data['Ingreso_total'])


def labels_encode(data:object) -> None:
	"""
		Transform string labels into numeric data suitable for ML model in place.

		Arguments:
			object data:
				Input dataframe to be transformed.

		Returns:
			None.
	"""

	encoder = sklearn.preprocessing.LabelEncoder()

	for i in range(0, 8):
		if i == 5:
			# Plazo de préstamo
			continue

		data[:,i] = encoder.fit_transform(data[:,i])


def model_input_get(data:object) -> object:
	"""
		Returns input data suitable for loan evaluation model.

		Arguments:
			object data:
				Pandas input dataframe to be filtered.

		Returns:
			Dataframe input suitable for loan evaluation model.
	"""

	return data[[
		'Genero', 'Casado', 'Dependientes', 'Educacion', 'Trabaja_para_el',
		'Plazo_prestamo', 'Historial_crediticio', 'Area_vivienda',
		'Credito_pedido_log', 'Ingreso_total_log'
	]].values
