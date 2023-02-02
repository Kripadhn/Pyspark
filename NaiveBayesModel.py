from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType

# Create a dataframe with sample data
data = [(1.0, Vectors.dense([0.0, 1.0, 2.0])),
(0.0, Vectors.dense([2.0, 3.0, 4.0])),
(1.0, Vectors.dense([4.0, 5.0, 6.0]))]
df = spark.createDataFrame(data, ["label", "features"])
df = df.withColumn("label", df["label"].cast(DoubleType()))

# Split the data into training and test sets
train, test = df.randomSplit([0.6, 0.4], seed=1)

# Create a NaiveBayes object and fit the model
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

# Predict the label for new data
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show(truncate=False)