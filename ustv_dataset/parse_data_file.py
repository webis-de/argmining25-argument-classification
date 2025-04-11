

import json
import settings as s
from collections import defaultdict

with open(s.PROJECT_ROOT/ "ustv_dataset/us2016.json", "r") as file:
    data = json.load(file)


scheme_dict = defaultdict(list)

for entry,arguments in data.items():
    if arguments["SCHEMES"] == []:
        continue
    for arg in arguments["FULL_ARGS"]:
        scheme = arg["scheme"]
        argument = arg["propositions"]
        scheme_dict[scheme].append(argument)

    print(arguments)

mewo = 1