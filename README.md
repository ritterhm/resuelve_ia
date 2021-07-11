# Evaluación de IA Resuelve tu Deuda

## Requirements

* Python 3.8
* Python 3 virtualenv package.

	```
		$ pip install virtualenv
	```
* Git source code manager.

If your python version differs from required you can use Anaconda or similar python package manager.

Virtual env is necessary to avoid polluted system wide common libraries.


## Installation (Linux)

1. Open shell terminal.

2. Download a copy from project.

	```
		$ git clone git@github.com:ritterhm/resuelve_ia.git
	```

3. Enter project directory.

	```
		$ cd $project_path
	```

4. Create a virtual environment for python.

	```
		$ python3 -m venv env
	```

5. Activate the virtual environment

	```
		$ source env/bin/activate
	```

6. Install needed libraries with `pip`.
	```
		$ pip install -r requirements.txt
	```

7. Do a test run.

	```
		$ cd release
		$ python model.py

		0    LP002519                 1
		1    LP001280                 1
		2    LP001151                 1
		3    LP002036                 1
		4    LP002894                 1
		..        ...               ...
		552  LP002894                 1
		553  LP002315                 0
		554  LP002225                 1
		555  LP002807                 1
		556  LP001664                 1
	```

8. Exit virtualenv.

	```
		$ deactivate
	```

## Model Class and Libraries

### src/common.py

This library has all common functions to make enable model to run. Be sure you
copy this file to `$project_path/release/` if you do changes in it.

### src/model_build.py

Final model serializer it always output to `$project_path/release/model.ml`. Ensure run it if you make changes on it.

1. Open shell terminal.

2. Enter project `src` directory.

	```
		$ cd $project_path/src/
	```

3. Activate the virtual environment

	```
		$ source env/bin/activate
	```

4. Rebuild model

	```
		$ python model_build.py

		[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
		 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 0 0 1 1 1 0 1 1 1 1 1 1 1 1 1
		 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 1 1
		 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
		 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1
		 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
		 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1
		 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
		Accuracy:  0.8530465949820788
	```

9. Exit virtualenv.

	```
		$ deactivate
	```

### src/model.py

Model release. Make sure copy it to. This file It also capable of self run as
command. Be sure you copy this file to `$project_path/release/` if you do
changes in it.

#### LoanEvaluator class

This class only have a constructor and evaluator method. It can be used as component inside a larger system.

To instantiate class do the follow you need a preserialized model file (see `src/model_build.py` section):
	```
		model = LoanEvaluator(serialized_model_path: str)
	```

To evaluate loan request to be elegible use `evaluate` method. It accepts as argument a Pandas dataframe with same data as the BI evaluation states. It returns a Pandas dataframe with Request I and loan elegible result: 1 for elegible and 0 for non elegible.

	```
		LoanEvaluator.evaluate(data: dataframe) -> dataframe
	```

Example of use:

	```
		data = pandas.read_csv('../data/datos_prestamo.csv')
		model = LoanEvaluator('../release/model.ml')
		model.evaluate(data)
	```

## Production Like Deploying

To deploy model in a production like environment, install the follow python libraries according your internal policy:

* numpy
* pandas
* sklearn

Ensure library `release/common.py` is reachable from `release/model.py` and install according your internal policy.

Is also necessary to make reachable preserialized model `release/model.ml` for `release/model.py` file.

You can use instances of `LoanEvaluator` class as component in major flow.
