#!/usr/bin env python

# ===============================================================================
# Model for evaluating loans.
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
#import numpy
import pandas
#import sklearn.linear_model
#import sklearn.model_selection
import sklearn.preprocessing

# Project libraries
import common


# -----------------------------------------------------------------------------
# Model classs
# -----------------------------------------------------------------------------

class LoanEvaluator:
	def __init__(self, model_path:str) -> None:
		"""
			Initializes LoanEvaluator.

			Arguments:
				Precompiled model file path.

			Returns:
				None
		"""

		self.model = joblib.load(model_path)

	def evaluate(self, data:object) -> object:
		"""
			Evaluates loans data for loan eligibility.

			Arguments:
				dataframe data:
					Pandas dataframe with loan input parameters.

			Returns
				Pandas dataframe with loan request id and loan eligibility
				status. If value is 1 this row is loan elegible all else means
				not elegible for.
		"""

		data = data.copy()
		scaler = sklearn.preprocessing.StandardScaler()

		common.data_transform(data)
		input_data = common.model_input_get(data)
		common.labels_encode(input_data)
		input_data = scaler.fit_transform(input_data)
		prediction = self.model.predict(input_data)

		output = data[['Id']]
		output['Estatus_prestamo'] = prediction

		return output


# -----------------------------------------------------------------------------
# Main Code
# -----------------------------------------------------------------------------

if __name__ == '__main__':
	data = pandas.read_csv('../data/datos_prestamo.csv')
	model = LoanEvaluator('../release/model.ml')
	print(model.evaluate(data))
