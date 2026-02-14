import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract,collect_list, concat_ws ,col,count,desc,length, avg,stddev,to_date


#spark =  SparkSession.builder.appName("assignment1").getOrCreate()
spark = SparkSession.builder \
    .appName("assignment1") \
    .config("spark.local.dir", "/home/suvendu/spark-temp") \
    .getOrCreate()

#df_spark = spark.read.option('header','true').text('/home/suvendu/mlbd/D184MB/200.txt')


books_df = spark.read.text("/home/suvendu/mlbd/D184MB/*.txt") \
    .withColumn("file_path", input_file_name()) \
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+$)", 1)) \
    .groupBy("file_name") \
    .agg(concat_ws(" ", collect_list("value")).alias("text"))

books_df.printSchema()
#books_df.show(5, truncate=False)

#10. Book Metadata Extraction and Analysis
'''Task:
    1. Metadata Extraction: Extract the following metadata from the text column of each
book in the books_df DataFrame:
- title
- release_date
- language
- encoding 
example 
Title: Pride and Prejudice
Release Date: August 26, 2008 [EBook #1342]
Language: English
Character set encoding: UTF-8
we have schema :
file_name (string)
text (string)
we will extract metadata using regexp_extract() from the text column.
'''
books_df.select("file_name").show(5)

books_with_title = books_df.withColumn(
    "title",
    regexp_extract(col("text"), r"(?i)Title:\s*(.*)", 1)
)
print(books_with_title.printSchema())

books_with_date = books_with_title.withColumn(
    "release_date",
    regexp_extract(col("text"), r"(?i)Release Date:\s*(.*)", 1)
)

print(books_with_date.printSchema())

books_with_language = books_with_date.withColumn(
    "language",
    regexp_extract(col("text"), r"(?i)Language:\s*(.*)", 1)
)
print(books_with_language.printSchema())

books_with_encoding = books_with_language.withColumn(
    "encoding",
    regexp_extract(col("text"), r"(?i)(Character set encoding|Encoding):\s*(.*)", 2)
)


#books_with_title.select("file_name", "title").show(5, truncate=False)

#books_with_date.select("file_name", "release_date").show(5, truncate=False)

#books_with_language.select("file_name", "language").show(5, truncate=False)

#books_with_encoding.select("file_name", "encoding").show(5, truncate=False)


'''2. Analysis:
- Calculate the number of books released each year.
- Find the most common language in the dataset.
- Determine the average length of book titles (in characters).'''

#Number of Books Released Each Year
#Step 1 — Extract Year from release_date

books_with_year = books_with_encoding.withColumn(
    "year",
    regexp_extract(col("release_date"), r"(\d{4})", 1)
)

books_with_year.show()

'''Explanation:

\d{4} → exactly 4 digits

( ) → capture group

1 → extract first group'''

#Step 2 — Count books per year


books_per_year = books_with_year.groupBy("year") \
    .agg(count("*").alias("num_books")) \
    .orderBy("year")

books_per_year.show()

# Find the most common language in the dataset


language_count = books_with_encoding.groupBy("language") \
    .agg(count("*").alias("num_books")) \
    .orderBy(desc("num_books"))


language_count.show(5)
#The top row = most common language.

#Average Length of Book Titles
#Step 1 — Calculate title length


books_with_title_length = books_with_encoding.withColumn(
    "title_length",
    length(col("title"))
)

#Step 2 — Calculate average

avg_title_length = books_with_title_length.select(
    avg("title_length")
)

avg_title_length.show(5)


spark.stop()