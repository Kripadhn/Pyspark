from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

# Create a dataframe with sample data
data = [(Vectors.dense([0.0, 1.0, 2.0, 3.0, 4.0]),),
(Vectors.dense([2.0, 3.0, 4.0, 5.0, 6.0]),),
(Vectors.dense([4.0, 5.0, 6.0, 7.0, 8.0]),)]
df = spark.createDataFrame(data, ["features"])

# Create a PCA object and fit the model
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

# Transform the data to new principal components
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
