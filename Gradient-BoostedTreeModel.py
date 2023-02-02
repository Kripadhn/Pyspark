from pyspark.ml.classification import GBTClassifier
from pyspark.ml.linalg import Vectors

# Create a dataframe with sample data
data = [(Vectors.dense([0.0, 1.0]), 0.0),
        (Vectors.dense([2.0, 3.0]), 1.0),
        (Vectors.dense([4.0, 5.0]), 0.0)]
df = spark.createDataFrame(data, ["features", "label"])

# Create a gradient-boosted tree object and fit the model
gbt = GBTClassifier(maxIter=5, maxDepth=2)
model = gbt.fit(df)

# Predict the output for new data
test = spark.createDataFrame([(Vectors.dense([5.0, 6.0]),)], ["features"])
result = model.transform(test).collect()[0]
print("Prediction:", result.prediction)
