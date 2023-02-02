from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import Vectors

# Create a dataframe with sample data
data = [(0, Vectors.dense([0.0, 1.0, 2.0])),
        (1, Vectors.dense([2.0, 3.0, 4.0])),
        (2, Vectors.dense([4.0, 5.0, 6.0]))]
df = spark.createDataFrame(data, ["id", "features"])

# Create an LDA object and fit the model
lda = LDA(k=2, seed=1)
model = lda.fit(df)

# Predict the topic for new data
test = spark.createDataFrame([(Vectors.dense([5.0, 6.0, 7.0]),)], ["features"])
result = model.transform(test).collect()[0]
print("Prediction:", result.topicDistribution)
