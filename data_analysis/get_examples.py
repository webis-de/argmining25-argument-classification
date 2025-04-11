import mongodb.mongo_handler as mdb
import settings as s
import model_settings as ms
import utils.utils as ut

ethix = mdb.get_data_from_mongo(collection_name=ms.ETHIX_SPLIT, filter_dict={ms.DATASET_NAME: ms.ETHIX_SPLIT, ms.SPLIT : ms.TRAIN, ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES_TOPICS})
utils_dict_ethix = ut.convert_argument_list_to_schemes_dict(ethix)



ustv = mdb.get_data_from_mongo(collection_name=ms.USTV2016_SPLIT, filter_dict={ms.DATASET_NAME: ms.USTV2016_SPLIT, ms.SPLIT : ms.TRAIN, ms.SPLIT_IDENTIFIER : ms.SPLIT_SCHEMES})
utils_dict_ustv = ut.convert_argument_list_to_schemes_dict(ustv)


mewo =1