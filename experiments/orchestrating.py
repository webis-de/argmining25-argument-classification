# use of multi-processing experiment to speed up the training process

# list of orchestrators, is used to run the experiment
# import multiprocessing as mp
import time

from loguru import logger
import mongodb.mongo_handler as mdb
import model_settings as ms
import settings as s
import utils.utils as ut

def classify_instance(instance) :  # helper for the pool map
    instance.classify_data()
    return instance

# INPUT: list of experiment classes, which can be run in parallel, therefore each experiment class is self contained
class Orchestrator():
    def __init__(self,experiment_classes_list,additional_fields_check_existence=None,CHECK_EXISTENCE_FLAG=True): # FLAG is used to determine if additional field checks should be performed
        super().__init__()
        self.experiment_list = experiment_classes_list

        self.check_existence = CHECK_EXISTENCE_FLAG
        self.additional_fields_check_existence = additional_fields_check_existence
        if additional_fields_check_existence is None:
            self.additional_fields_check_existence = []

        # if s.NUM_PROCESSES > 1:
        #     logger.info("Using multi processing")
        #     self.do_multi_classification()
        # else:
        logger.info("Using single processing")
        self.do_single_classifications()


    def check_experiment_data_instance_mongodb(self, data_instance):
        if not self.check_existence:
            return

        # Get the collection name from the data instance meta
        collection_name = data_instance.meta[ms.COLLECTION]

        # Construct argument_keys_in_use dictionary
        argument_keys_in_dataset = {ms.ARGUMENT_ID: [x[ms.ARGUMENT_ID] for x in data_instance.Dataset[ms.DATA]]}

        # Create MongoDB query filters based on the argument_keys_in_use, dict
        mongo_filters = {}
        for key, values in argument_keys_in_dataset.items():
            mongo_filters[key] = {"$in": values}


        for field in self.additional_fields_check_existence:
            mongo_filters[field] = data_instance.meta[field]

        # Query the MongoDB collection with the constructed filters
        existing_documents = mdb.get_data_from_mongo(collection_name,mongo_filters)

        # Extract the existing argument IDs from the documents
        existing_argument_ids = [doc[ms.ARGUMENT_ID] for doc in existing_documents]

        if len(existing_argument_ids) == 0:
            return

        # Log the number of existing argument IDs found
        logger.info(f"Found {len(existing_argument_ids)} existing IDs in the current batch of {len(argument_keys_in_dataset)} entries")

        # Filter out existing IDs for data instance
        filtered_dataset = [i for i in data_instance.Dataset[ms.DATA] if i[ms.ARGUMENT_ID] not in existing_argument_ids]
        data_instance.Dataset[ms.DATA] = filtered_dataset



    def upload_data_instances(self, data_instance):
        meta = data_instance.meta
        collection = meta[ms.COLLECTION]
        data = data_instance.storage_classified_schemes_list
        if len(data) == 0:
            return
        # not too many checks are needed, since there is a general check before the data is uploaded
        mdb.upload_data_to_mongo(collection_name=collection, batch_data=data, keys_to_check=None,meta_data=meta)

        # uploader.meta = data_instance.meta
        # uploader.upload_data_to_elastic(data_instance.storage_classified_schemes_list)


    def do_single_classifications(self): # do classification for each experiment in the data list
        for i,experiment in enumerate(self.experiment_list):
            self.check_experiment_data_instance_mongodb(experiment)
            experiment.classify_data()
            self.upload_data_instances(experiment)
            logger.info(f"Experiment {i+1} of {len(self.experiment_list)} done")


    ## utils needed for multi processing

    # def do_multi_classification(self):
    #     experiments_batch_list =  ut.split_list(self.experiment_list)
    #     for experiment_list in experiments_batch_list:
    #         for single_experiment_instance in experiment_list:
    #             self.check_experiment_data_instance_mongodb(single_experiment_instance)
    #
    #         classified_data = self._do_multi_classification_pool(experiment_list)
    #         for single_experiment_instance in classified_data:
    #             self.upload_data_instances(single_experiment_instance)
    #         logger.info(f"Experiment {len(experiment_list)} of {len(self.experiment_list)} done")
    #
    #
    #
    # def _do_multi_classification_pool(self, data_list) :
    #
    #     # Split the list into sublists for the processes
    #     final_data_list = []
    #     multi_instances_run_list = ut.split_list(data_list,
    #                                              s.NUM_PROCESSES)  # Each entry represents one run with multiple processes
    #     start_time = time.time()
    #     # Create a pool of processes
    #         # Process each chunk of text in parallel
    #     with mp.Pool(processes=s.NUM_PROCESSES) as pool :
    #         for instances in multi_instances_run_list :
    #             processed_chunks = pool.map(classify_instance, instances)
    #             final_data_list.extend(processed_chunks)  # Collect the processed chunks
    #
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time:.2f} seconds")
    #     return final_data_list
