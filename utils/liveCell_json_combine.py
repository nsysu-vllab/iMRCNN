import os
import csv
import glob
import json

# python3 utils/liveCell_json_combine.py

images = []
annos = []

for fname in glob.glob(os.path.join('/home/frank/Desktop/instance segmentation/SegPC-2021-main/data/AAA_json_combine_test/AAA_dataset_test', '*.json')):
    with open(fname) as f:
        jf = json.loads(f.read())
        images.extend(jf['images'])
        annos.extend(jf['annotations'])
        categories = jf['categories']
        
dataset = {
    "images": images,
    "annotations": annos,
    "categories": categories,
}

#with open('/home/frank/Desktop/instance segmentation/SegPC-2021-main/data/AAA_dataset_test/output.json', 'w') as f:
#    print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')), file=f)
with open('/home/frank/Desktop/instance segmentation/SegPC-2021-main/data/AAA_json_combine_test/AAA_dataset_test/output.json', 'w') as fp:
    json.dump(dataset, fp)
    print("good")

#with open('/home/frank/Desktop/instance segmentation/SegPC-2021-main/data/AAA_dataset_test/output.csv', 'w') as f:
#    writer = csv.writer(f)
#    writer.writerow([key for key, value in data[0].items()])
#    for d in data:
#        writer.writerow([value for key, value in d.items()])

