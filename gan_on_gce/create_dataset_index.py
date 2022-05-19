import json
import os
source = "./gans_training/images"
destination = "./gans_training/images"
files_list = os.listdir(source)
data_dict = {}
data_dict['labels'] = []
curr_class = None
class_counter = -1
for files in sorted(files_list):
  x = files.rsplit('_', 1)[0]
  if x != curr_class:
    curr_class = x
    class_counter += 1
  data_dict['labels'].append([files, class_counter])
with open(os.path.join(destination, 'dataset.json'), 'w') as outfile:
  json.dump(data_dict, outfile)
