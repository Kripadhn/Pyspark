from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

# Create a dataframe with sample data
data = [(Vectors.dense([0.0, 1.0]),),
        (Vectors.dense([2.0, 3.0]),),
        (Vectors.dense([4.0, 5.0]),)]
df = spark.createDataFrame(data, ["features"])

# Create a KMeans object and fit the model
kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(df)

# Predict the cluster for new data
test = spark.createDataFrame([(Vectors.dense([5.0, 6.0]),)], ["features"])
result = model.transform(test).collect()[0]
print("Prediction:", result.prediction)
