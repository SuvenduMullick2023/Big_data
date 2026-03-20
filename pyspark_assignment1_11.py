

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract,collect_list, concat_ws ,col,count,desc,length, avg,stddev,to_date


#spark =  SparkSession.builder.appName("assignment1").getOrCreate()
# Remove Gutenberg Header/Footer
'''spark = SparkSession.builder \
    .appName("assignment1") \
    .config("spark.local.dir", "/home/suvendu/spark-temp") \
    .getOrCreate()'''
    


spark = SparkSession.builder \
    .appName("TFIDF") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Reduce Shuffle Size to 4 from default 200
spark.conf.set("spark.sql.shuffle.partitions", "4")

#df_spark = spark.read.option('header','true').text('/home/suvendu/mlbd/D184MB/200.txt')


books1_df = spark.read.text("/home/suvendu/mlbd/D184MB/*.txt") \
    .withColumn("file_path", input_file_name()) \
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+$)", 1)) \
    .groupBy("file_name") \
    .agg(concat_ws(" ", collect_list("value")).alias("text"))

books1_df.printSchema()

books_df = books1_df.limit(50)


#Task: Calculate TF-IDF scores for words in each book and use them to determine book
#      similarity based on cosine similarity.


# BEFORE CLEANING — See Original Text (Small Part)
from pyspark.sql.functions import substring

books_df.select(
    "file_name",
    substring("text", 1, 100).alias("original_preview")
).show(truncate=False)



'''1. Preprocessing:
- Clean the text column in books_df : 
        remove Project Gutenberg header/footer,
        convert to lowercase, 
        remove punctuation, 
        tokenize into words, 
        and remove stop words.'''


'''Explanation:

(?s) → dot matches newline

.*? → non-greedy match

Removes header and footer blocks'''

from pyspark.sql.functions import regexp_replace, col
# Step 1 - Remove Gutenberg Header/Footer
clean_df = books_df.withColumn(
    "clean_text",
    regexp_replace(
        col("text"),
        r"(?s)\*\*\* START OF.*?\*\*\*|\*\*\* END OF.*?\*\*\*",
        ""
    )
)

'''clean_df.select(
    "file_name",
    substring("clean_text", 1, 100).alias("after_header_removal")
).show(truncate=False)'''



# Step 2 — Lowercase
from pyspark.sql.functions import lower

clean_df = clean_df.withColumn(
    "clean_text",
    lower(col("clean_text"))
)

# Step 3 — Remove Punctuation : This keeps only: letters a-z and  spaces

clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), "[^a-z\\s]", "")
)

# Step 4 — Tokenize into Words

from pyspark.sql.functions import split

tokenized_df = clean_df.withColumn(
    "words",
    split(col("clean_text"), "\\s+")
)

# Step 5 — Remove Stopwords : Use Spark ML stopwords remover.

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

filtered_df = remover.transform(tokenized_df)

filtered_df.select(
    "file_name",
    substring("text", 1, 100).alias("StopWordsRemover")
).show(truncate=False)



'''2.TF-IDF Calculation:
- Calculate the Term Frequency (TF) of each word in each book.
- Calculate the Inverse Document Frequency (IDF) for each word across all books.
- Compute the TF-IDF score for each word in each book (TF * IDF).'''

# TF (Term Frequency)  = (number of times word appears in book)

# Step 6 — Explode Words

from pyspark.sql.functions import explode

words_exploded = filtered_df.select(
    "file_name",
    explode(col("filtered_words")).alias("word")
)

# Step 7 — Count Word Occurrences Per Book

from pyspark.sql.functions import count

tf_df = words_exploded.groupBy("file_name", "word") \
    .agg(count("*").alias("tf"))


# PART 3 — IDF (Inverse Document Frequency)
# IDF=log(TotalBooks/NumberOfBooksContainingWord)

# Step 1 — Total Number of Books
total_books = books_df.count()

#Step 2 — Count Documents Containing Each Word
from pyspark.sql.functions import countDistinct

doc_freq_df = words_exploded.groupBy("word") \
    .agg(countDistinct("file_name").alias("doc_count"))

del words_exploded

# Step 10 — Calculate IDF
from pyspark.sql.functions import log

idf_df = doc_freq_df.withColumn(
    "idf",
    log(total_books / col("doc_count"))
)


# PART 4 — Compute TF-IDF
#TF-IDF = TF × IDF

# Step 1 — Join TF and IDF
tfidf_df = tf_df.join(idf_df, on="word")

# Step 2 — Calculate TF-IDF Score

tfidf_df = tfidf_df.withColumn(
    "tfidf",
    col("tf") * col("idf")
)

# View Top Important Words Per Book
from pyspark.sql.functions import desc

top_words_per_book = tfidf_df.orderBy(desc("tfidf"))

top_words_per_book.show(20)

tfidf_df.unpersist()
del tfidf_df
spark.catalog.clearCache()



'''Book Similarity:
- Represent each book as a vector of its TF-IDF scores.
- Calculate the cosine similarity between all pairs of book vectors.
- For a given book (e.g., file_name "10.txt"), identify the top 5 most similar books based
on cosine similarity.
'''

# Convert Words → Term Frequency Vector

from pyspark.ml.feature import HashingTF

hashingTF = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_features",
    numFeatures=10000   # you can adjust
)

featurized_df = hashingTF.transform(filtered_df)

# Compute IDF
from pyspark.ml.feature import IDF

idf = IDF(inputCol="raw_features", outputCol="features")

idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# STEP 2 — Cosine Similarity
# Normalize Vectors

from pyspark.ml.feature import Normalizer

normalizer = Normalizer(inputCol="features", outputCol="norm_features")

norm_df = normalizer.transform(tfidf_df)

# STEP 3 — Self Join to Compare All Books
# Self Join

df1 = norm_df.alias("df1")
df2 = norm_df.alias("df2")

similarity_df = df1.join(df2, col("df1.file_name") != col("df2.file_name"))

# Compute Cosine Similarity
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors

def cosine_similarity(v1, v2):
    return float(v1.dot(v2))

cosine_udf = udf(cosine_similarity, DoubleType())

similarity_df = similarity_df.withColumn(
    "similarity",
    cosine_udf(col("df1.norm_features"), col("df2.norm_features"))
)

# STEP 4 — Top 5 Similar Books to "10.txt"

from pyspark.sql.functions import desc

top5 = similarity_df \
    .filter(col("df1.file_name") == "10.txt") \
    .select(
        col("df2.file_name").alias("similar_book"),
        "similarity"
    ) \
    .orderBy(desc("similarity")) \
    .limit(5)

top5.show()

# Method 2 ###############################################################
from pyspark.ml.feature import BucketedRandomProjectionLSH

lsh = BucketedRandomProjectionLSH(
    inputCol="norm_features",   # vector column
    outputCol="hashes",
    bucketLength=1.5,           # controls similarity sensitivity
    numHashTables=3             # more tables = better accuracy, more compute
)

lsh_model = lsh.fit(norm_df)
hashed_df = lsh_model.transform(norm_df)
query_book = norm_df.filter(col("file_name") == "10.txt")
similar_books = lsh_model.approxNearestNeighbors(
    dataset=norm_df,
    key=query_book.select("norm_features").first()["norm_features"],
    numNearestNeighbors=6   # include itself
)
from pyspark.sql.functions import col

top5 = similar_books \
    .filter(col("file_name") != "10.txt") \
    .select("file_name", "distCol") \
    .orderBy("distCol") \
    .limit(50)

top5.show(20)

#  Compare All Books Pairwise
similarity_pairs = lsh_model.approxSimilarityJoin(
    norm_df,
    norm_df,
    threshold=1.5,
    distCol="distance"
)

similarity_pairs.show(20)
spark.stop()