from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors

# Create a dataframe with sample data
data = [(Vectors.dense([0.0, 1.0]), 1.0),
        (Vectors.dense([2.0, 3.0]), 2.0),
        (Vectors.dense([4.0, 5.0]), 3.0)]
df = spark.createDataFrame(data, ["features", "label"])

# Create a linear regression object and fit the model
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
model = lr.fit(df)

# Predict the output for new data
test = spark.createDataFrame([(Vectors.dense([5.0, 6.0]),)], ["features"])
result = model.transform(test).collect()[0]
print("Prediction:", result.prediction)
