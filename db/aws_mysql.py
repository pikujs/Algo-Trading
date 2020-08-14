import pymysql
import pandas as pd
from sqlalchemy import create_engine

print("Connecting")
conn = pymysql.connect(host='tick-data.ckgtm0hu3fjh.ap-south-1.rds.amazonaws.com', port=1433, user='admin', password='admin ', database='datastore')
print("conneccted")
cursor = conn.cursor()
print("Got cursor")

query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA='datastore'"
r = cursor.execute(query)

for item in r:
    print(item)


"""
result = pd.read_sql(query, cursor)
print(result.head())


cursor.execute(query)
with open("output.csv","w") as outfile:
    writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(col[0] for col in cursor.description)
    for row in cursor:
        writer.writerow(row)

query = 'select * from datastore'
results = pd.read_sql_query(query, conn)
print("pd read")
results.to_csv("test_output.csv", index=False)
print("file saved")
"""