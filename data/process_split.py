import os
import json

split = 'val' # {train, val}
with open('C:\\Users\\zyang\\Downloads\\splits\\{}_models.json'.format(split), 'r') as f:
  data = json.load(f)

for category in data.keys():
  print('process category {}'.format(category))
  if category in ['03001627', '02691156', '02958343']:
    continue
  with open('D:\\data\\images\\data\\splits\\{}_{}_list.txt'.format(category, split), 'w') as f:
    for file in data[category]:
      f.write(file.split('/')[1] + '\n')