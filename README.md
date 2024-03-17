# Representativeness model

Author: Wojciech Konarzewski (konarz48@gmail.com)

## Model

The model returns a representativeness of an object in the form of a number.

Each object (in the training dataset and later when making predictions) should be represented by a list of numbers (e.g. ```[54.0,3,6,10,1968,52.1898,21.0159,12778,1.91]```).

Representativeness model creation process:
1. The dataset is randomly divided into **L** parts with equal number of objects
2. For each dataset:
    1. Representativeness score for each object is calculated in steps listed below:
        1. The distance to **K** nearest neighbors of an object is calculated using [sklearn.neighbors.NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) model and ```Minkowski distance``` metric.
        2. Mean of those distances is calculated.
        3. Representativeness score is then calculated using the formula ```1/(1+mean_distance)```
    2. [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) model is trained. Calculated representativeness scores are used as targed variable.
3. The final model is an ensemble consisting of all trained individual models. The predictions returned by the final model are the average of the predictions of the individual models.


## API

Production service url: https://representativeness-model.onrender.com

Production auth token: ---

---

**CAUTION** 

The current project configuration involves deleting trained models and resetting the training status when shutting down the application. In addition, it is worth mentioning that the production service has been deployed on the [render.com](render.com) platform on the Free instance type. This implies certain limitations, which are described in the [documentation](https://docs.render.com/free#spinning-down-on-idle):

> Render spins down a Free web service that goes 15 minutes without receiving inbound traffic. Render spins the service back up whenever it next receives a request to process. Spinning up a service takes a few seconds, which causes a noticeable delay for incoming requests until the service is back up and running.

However, my experience has shown that the delay can be even more than a minute. If you are unable to use the service after such time, please contact me (I will manually restart the application).

The service may also not be able to cope with large datasets for model training (the free instance does not have impressive resources). In such a situation, I recommend running the app locally (instructions below).

---

The API consists of 3 main endpoints:

1. **POST** ```/train``` (starts new ensemble model training)

Input:
- n_split (int, default=1): indicates how many parts the dataset should be divided into (parameter **L** in model description above)
- n_nearest (int, default=2): number of nearest neighbors taken into account (parameter **K** in model description above)
- data (array or objects): training dataset

Example request:
```
curl -X POST -H "Auth-Token: <AUTH TOKEN>" -H "Content-Type: application/json" -d '{"n_split":5,"n_nearest":20, "data":[[...],[...],[...],...]}' <SERVICE URL>/train
```

Example response:
```
{
	"details": "New training started",
	"in_progress": true,
	"start_time": "2024-03-16T21:24:30.116226"
}
```
If the started training has finished, the next request will start a new training. If the training has not finished, the new training will not start and the response will look  like this:
```
{
	"details": "Training in progress",
	"in_progress": true,
	"start_time": "2024-03-16T21:24:30.116226"
}
```

2. **GET** ```/status``` (checks model training status)

Example request:
```
curl -X POST -H "Auth-Token: <AUTH TOKEN>" -H "Content-Type: application/json" <SERVICE URL>/status
```

Responses:

- Before first training
```
{
	"details": "No training recorded so far",
	"in_progress": false
}
```
- Training in progress
```
{
	"details": "Training in progress",
	"in_progress": true,
	"start_time": "2024-03-16T21:27:32.355437"
}
```
- Training ended
```
{
	"details": "Training successfully completed",
	"in_progress": false,
	"prod_model": "5850cf4ed87d232d5fc6e61612b737951140e31b0e36bb23dbc84450c9ffd1a4",
	"start_time": "2024-03-16T21:27:32.355437",
	"end_time": "2024-03-16T21:27:44.708929"
}
```
- Error during training
```
{
	"details": "Error occurred during training",
	"in_progress": false,
	"start_time": "2024-03-16T21:32:50.426234",
	"error_time": "2024-03-16T21:32:52.506893",
	"error": "Provided data could not be converted to numeric format"
}
```

3. **POST** ```/predict``` (make prediction)

Input:
- data (array or objects): object to make predictions for

Example request:
```
curl -X POST -H "Auth-Token: <AUTH TOKEN>" -H "Content-Type: application/json" -d '{"data":[[...],[...],[...]]}' <SERVICE URL>/predict
```

Example response for reuqest with 3 objects:
```
{
	"model": "0132db62d9c8b85ad06210a4b7237dfe3f24b7828389e3fbdac484c37a42744d",
	"prediction": [
		0.005539838458564862,
		0.001208184086673582,
		0.0055400689778954135
	]
}
```
If there has been no model trained yet, the response would look like:
```
{
	"details": "No models trained yet"
}
```

## How to run it locally?

1. Prerequisites
- installed git (used version: ```git version 2.34.1```)
- installed docker (used version: ```Docker version 25.0.3, build 4debf41```)
- installed docker-compose (used version: ```Docker Compose version v2.24.2```)

2. Clone this repository
```
git clone https://github.com/konarzewsky/Representativeness-model.git
cd Representativeness-model/
```

3. Build application
```
docker-compose build
```

4. Run application
```
docker-compose up
```

Local service url: http://0.0.0.0:5000

Dev auth token: ```dev_auth_token``` (can be changed in ```.env``` file)
