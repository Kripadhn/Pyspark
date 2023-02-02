from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import Vectors

# Create a dataframe with sample data
data = [(Vectors.dense([0.0, 1.0]), 0.0),
        (Vectors.dense([2.0, 3.0]), 1.0),
        (Vectors.dense([4.0, 5.0]), 0.0)]
df = spark.createDataFrame(data, ["features", "label"])

# Create a random forest object and fit the model
rf = RandomForestClassifier(numTrees=3, maxDepth=2)
model = rf.fit(df)

# Predict the output for new data
test = spark.createDataFrame([(Vectors.dense([5.0, 6.0]),)], ["features"])
result = model.transform(test).collect()[0]
print("Prediction:", result.prediction)
