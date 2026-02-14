import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract,collect_list, concat_ws ,col,count,desc,length, avg,stddev,to_date


#spark =  SparkSession.builder.appName("assignment1").getOrCreate()

    


spark = SparkSession.builder \
    .appName("TFIDF") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Reduce Shuffle Size to 4 from default 200
spark.conf.set("spark.sql.shuffle.partitions", "4")

#df_spark = spark.read.option('header','true').text('/home/suvendu/mlbd/D184MB/200.txt')


'''books_df = spark.read.text("/home/suvendu/mlbd/D184MB/*.txt") \
    .withColumn("file_path", input_file_name()) \
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+$)", 1)) \
    .groupBy("file_name") \
    .agg(concat_ws(" ", collect_list("value")).alias("text"))'''


from pyspark.sql.functions import input_file_name, regexp_extract, monotonically_increasing_id
from pyspark.sql.functions import concat_ws, collect_list, col
from pyspark.sql import Window
from pyspark.sql.functions import row_number

# Step 1: Read text
raw_df = spark.read.text("/home/suvendu/mlbd/D184MB/*.txt") \
    .withColumn("file_path", input_file_name()) \
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+$)", 1))

# Step 2: Preserve order inside each file
window = Window.partitionBy("file_name").orderBy(monotonically_increasing_id())

ordered_df = raw_df.withColumn(
    "line_id",
    row_number().over(window)
)

# Step 3: Reassemble text in correct order
books_df = ordered_df.groupBy("file_name") \
    .agg(concat_ws(" ", collect_list("value")).alias("text"))


books_df.printSchema()

#books_df = books1_df.limit(100)


#Task: Calculate TF-IDF scores for words in each book and use them to determine book
#      similarity based on cosine similarity.


# BEFORE CLEANING — See Original Text (Small Part)
from pyspark.sql.functions import substring

'''books_df.select(
    "file_name",
    substring("text", 1, 1000).alias("original_preview")
)''' #.show(truncate=False)




from pyspark.sql.functions import regexp_replace, col
# Step 1 - Extract Author


'''books_meta = books_df.withColumn(
    "author",
    regexp_extract(col("text"), r"(?i)Author:\s*(.*)", 1)
)'''

from pyspark.sql.functions import regexp_extract, col

books_meta = books_df.withColumn(
    "author",
    regexp_extract(col("text"), r"by\s+([A-Za-z.\s]+)", 1)
)



# Step 2 —Extract Release Date

books_meta = books_meta.withColumn(
    "year_str",
    regexp_extract(col("text"), r"(19\d{2}|20\d{2})", 1)
)

books_meta = books_meta.withColumn(
    "year",
    col("year_str").cast("int")
)




# Step 3 — Clean Nulls
books_meta = books_meta.filter(
    (col("author") != "") &
    (col("year").isNotNull())
)

books_meta.show(20)
# PART 2 — Influence Network Construction

'''We define:

Author A influences Author B
if B released within X years after A  let x =3 '''

# Step 4 — Remove Duplicate Authors (If Multiple Books)
# If an author has multiple books, we take earliest year.

from pyspark.sql.functions import min

authors_df = books_meta.groupBy("author") \
    .agg(min("year").alias("first_year"))

# Step 5 — Self Join Authors
from pyspark.sql.functions import abs

X = 5

a1 = authors_df.alias("a1")
a2 = authors_df.alias("a2")

edges_df = a1.join(
    a2,
    (col("a1.author") != col("a2.author")) &
    (col("a2.first_year") > col("a1.first_year")) &
    (col("a2.first_year") <= col("a1.first_year") + X)
).select(
    col("a1.author").alias("author1"),
    col("a2.author").alias("author2")
)

'''Meaning: created edges: (author1, author2)

author1 potentially influenced author2

Because:

author2 published after author1

within 3 years'''

# PART 3 — Network Analysis
# Out-Degree  = number of authors influenced
from pyspark.sql.functions import count

out_degree = edges_df.groupBy("author1") \
    .agg(count("*").alias("out_degree"))

# In-degree = number of authors who influenced them
in_degree = edges_df.groupBy("author2") \
    .agg(count("*").alias("in_degree"))

# Top 5 Highest Out-Degree
from pyspark.sql.functions import desc

top_out = out_degree.orderBy(desc("out_degree")).limit(5)

top_out.show()

# Top 5 Highest In-Degree
top_in = in_degree.orderBy(desc("in_degree")).limit(5)

top_in.show()


# Optional: Combine Degrees
author_degree = authors_df \
    .join(out_degree, authors_df.author == out_degree.author1, "left") \
    .join(in_degree, authors_df.author == in_degree.author2, "left") \
    .select(
        authors_df.author,
        "first_year",
        "in_degree",
        "out_degree"
    )

author_degree.show()


'''from graphframes import GraphFrame

spark = SparkSession.builder \
    .appName("InfluenceNetwork") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .getOrCreate()

vertices = authors_df.select(
    col("author").alias("id"),
    col("first_year")
)
edges = edges_df.select(
    col("author1").alias("src"),
    col("author2").alias("dst")
)
g = GraphFrame(vertices, edges)

in_deg = g.inDegrees
in_deg.orderBy(desc("inDegree")).show(5)

out_deg = g.outDegrees
out_deg.orderBy(desc("outDegree")).show(5)

g.degrees.orderBy(desc("degree")).show(5)

# PART 4 — Advanced Graph Analysis
results = g.pageRank(resetProbability=0.15, maxIter=10)

results.vertices \
    .select("id", "pagerank") \
    .orderBy(desc("pagerank")) \
    .show(5)

g.connectedComponents().show()
edges_pd = edges.toPandas()
vertices_pd = vertices.toPandas()
import networkx as nx
import matplotlib.pyplot as plt

G = nx.from_pandas_edgelist(
    edges_pd,
    source="src",
    target="dst",
    create_using=nx.DiGraph()
)

plt.figure(figsize=(12, 10))

pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=500,
    font_size=8,
    arrows=True
)

plt.title("Author Influence Network")
plt.show()
pagerank_df = results.vertices.toPandas()

pagerank_dict = dict(zip(pagerank_df["id"], pagerank_df["pagerank"]))

node_sizes = [pagerank_dict[node]*5000 for node in G.nodes()]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=node_sizes,
    arrows=True
)'''

spark.stop()