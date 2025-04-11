from pymongo import MongoClient
import settings as s
# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # adjust URI as needed

# Delete the database
db_name = client[s.MONGO_DB_NAME]
client.drop_database(db_name)

print(f"Database '{db_name}' deleted successfully.")