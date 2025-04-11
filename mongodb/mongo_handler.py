from pymongo import MongoClient, UpdateOne

import model_settings as ms
import settings as s

# Replace with your connection string
client = MongoClient("mongodb://localhost:27017/")  # or your MongoDB URI

# Select your database and collection
db = client[s.MONGO_DB_NAME]


def get_data_from_mongo(collection_name=None, filter_dict={}):
    filter_dict = filter_dict or {}

    if collection_name is None:
        if ms.COLLECTION in filter_dict:
            collection_name = filter_dict[ms.COLLECTION]
        else:
            raise ValueError(f"Collection name is not specified in the filter dict: {filter_dict}")

    collection = db[collection_name]
    # Check if the collection exists in the database
    if collection_name not in db.list_collection_names() :
        print(f"Collection '{collection_name}' does not exist in the database. Returning empty list.")
        return []

    # Check if keys in filter_dict exist in the collection
    for key in filter_dict.keys() :
        if not collection.find_one({key : {"$exists" : True}}) :
            raise KeyError(f"Key '{key}' does not exist in the collection '{collection_name}'.")

    results = list(collection.find(filter_dict))  # MongoDB uses implicit AND for filter dict
    if len(results) == 0:
        print(f"No data found in collection '{collection_name}' with filter {filter_dict}. Returning empty list.")
        return []
    return results


def upload_data_to_mongo(collection_name = None, batch_data = None, keys_to_check = None,meta_data=None):



    collection = db[collection_name]

    if meta_data is not None:
        for doc in batch_data:
            meta_data_copy = meta_data.copy()
            doc.update(meta_data_copy)

    if keys_to_check is None:
        result = collection.insert_many(batch_data)
        print(f"Inserted: {len(result.inserted_ids)} documents.")

    else:

        if not isinstance(keys_to_check, list):
            keys_to_check = [keys_to_check]

        operations = []
        for doc in batch_data :
            # Build filter from composite keys
            filter_criteria = {key : doc[key] for key in keys_to_check}
            operations.append(
                UpdateOne(
                    filter_criteria,
                    {"$setOnInsert" : doc},
                    upsert=True
                )
            )

        result = collection.bulk_write(operations)

        print(f"Inserted: {result.upserted_count}")
        print(f"Matched existing: {result.matched_count}")


if __name__ == "__main__":

    # Example usage
    collection_name = "testing_values"

    data_sample1 = {"field1": "value21", "field2": "value2", "field3": 1, "field4": None, "argument-id" : "6" }
    data_sample2 = {"field1": "value1", "field2": "value2", "field3": 1, "field4": None, "argument-id" : "2" }
    data_sample3 = {"field1": "value23", "field2": "value2", "field3": 1, "field4": None, "argument-id" : "2" }

    batch_data = [data_sample1, data_sample2, data_sample3]

    # Get data from MongoDB
    data = upload_data_to_mongo(collection_name, batch_data, keys_to_check="argument-id")
    print(data)

    # Upload data to MongoDB
    #batch_data = [{"field": "value1"}, {"field": "value2"}]
    #pload_data_to_mongo(collection_name, batch_data, id_to_check="field")
