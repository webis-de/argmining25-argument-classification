import data_handling.data_loader as dl
import model_settings as ms

ethix_data = dl.load_ethix_data()
all_topics = list(set([x[ms.TOPIC] for x in ethix_data]))
all_schemes = list(set([x[ms.SCHEME] for x in ethix_data]))


# split the data lists into topics and schemes
def _split_arguments_into_topics(argument_list):
    topic_dict = {topic : [] for topic in all_topics}
    for argument in argument_list:
        topic_dict[argument[ms.TOPIC]].append(argument)
    return topic_dict

def _split_arguments_into_schemes(argument_list):
    schemes_dict = {scheme : [] for scheme in all_schemes}
    for argument in argument_list:
        schemes_dict[argument[ms.SCHEME]].append(argument)
    return schemes_dict

def create_nested_scheme_dict(argument_list):
    scheme_topic_dict = {}
    tmp_dict = _split_arguments_into_schemes(argument_list)
    for scheme, arguments in tmp_dict.items():
        scheme_topic_dict[scheme] = _split_arguments_into_topics(arguments)
    return scheme_topic_dict


if __name__ == "__main__":

    # Create nested dictionary of schemes and their topics
    nested_dict = create_nested_scheme_dict(ethix_data)

    # For each scheme, analyze topic distribution
    for scheme, topic_dict in nested_dict.items():
        print(f"\nScheme: {scheme}")
        print("-" * 50)
        
        # Count total arguments for this scheme
        total_args = sum(len(args) for args in topic_dict.values())
        
        # Calculate and display frequency distribution
        print("Topic frequency distribution:")
        # Sort topics by frequency in descending order
        sorted_topics = sorted(topic_dict.items(), key=lambda x: len(x[1]), reverse=True)
        for topic, arguments in sorted_topics:
            count = len(arguments)
            percentage = (count / total_args) * 100 if total_args > 0 else 0
            print(f"{topic}: {count} arguments ({percentage:.1f}%)")