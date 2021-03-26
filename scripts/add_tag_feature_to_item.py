import csv

originalMLdatasetPath = None
recboleMLdatasetPath = None
infile = f"{originalMLdatasetPath}/ml-20m/tags.csv"
dataset = "ml-100k"
itemfile = f"{recboleMLdatasetPath}/{dataset}/{dataset}.item"

tags = {}
with open(infile, 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for line in reader:
        movieid = line[1].strip()
        if movieid not in tags:
            tags[movieid] = []
        tags[movieid].append(line[2].strip())

items = []
with open(itemfile, 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    header = next(reader)
    for line in reader:
        items.append(line)

with open(itemfile, 'w') as fout:
    writer = csv.writer(fout, delimiter='\t')
    header.append("tags:token_seq")
    writer.writerow(header)
    for item in items:
        item_id = item[0]
        if item_id in tags:
            item.append(" ".join(tags[item_id]))
        else:
            print(item_id)
            # item.append("NONE")
            item.append("")
        writer.writerow(item)
