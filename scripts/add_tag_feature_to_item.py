import csv
import re

originalMLdatasetPath = None
recboleMLdatasetPath = None
infileTags = f"{originalMLdatasetPath}/ml-20m/tags.csv"
infileMovies = f"{originalMLdatasetPath}/ml-20m/movies.csv"
dataset = "ml-100k"
original_itemfile = f"{recboleMLdatasetPath}/{dataset}/{dataset}.originalitems"
new_itemfile = f"{recboleMLdatasetPath}/{dataset}/{dataset}.item"

tags_id = {}
with open(infileTags, 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for line in reader:
        movieid = line[1].strip()
        if movieid not in tags_id:
            tags_id[movieid] = []
        tags_id[movieid].append(line[2].strip())

title_id_mapping_20ml = {}
with open(infileMovies, 'r') as fin:
    reader = csv.reader(fin)
    next(reader)
    for line in reader:
        title = line[1].strip().lower()
        title_id_mapping_20ml[title] = line[0].strip()
        title2 = re.sub(r'^([^(]+)(\(.*\))? (\(\d\d\d\d\))$', r'\g<1>\g<3>', title)
        title_id_mapping_20ml[title2] = line[0].strip()
        # if title2 != title:
        #     print(f"{title} -> {title2}")

items = []
with open(original_itemfile, 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    header = next(reader)
    for line in reader:
        items.append(line)

with open(new_itemfile, 'w') as fout:
    writer = csv.writer(fout, delimiter='\t')
    header.append("tags:token_seq")
    writer.writerow(header)
    for item in items:
        item_id = item[0].strip()
        item_title = item[1].strip().lower()
        item_title = re.sub(r'^([^(]+)( \(.*\))?$', r'\g<1>', item_title)
        if item[2].strip() in ['unkonwn']:
            print(item)
            continue
        if item[2].strip() == "V":
            item_title = 'Land Before Time III: The Time of the Great Giving'
            item_year = 1995
        else:
            item_year = int(item[2].strip())
        found = False
        for item_ty in [item_title + f" ({item_year})", item_title + f" ({item_year+1})", item_title + f" ({item_year-1})"]:
            if item_ty in title_id_mapping_20ml:
                ml20_id = title_id_mapping_20ml[item_ty]
                if dataset == "ml-20m":
                    if ml20_id != item_id:
                        print(f"something is wrong! {ml20_id} != {item}")
                        exit()
                if ml20_id in tags_id:
                    item.append(" ".join(tags_id[ml20_id]))
                    found = True
                    break
        if found is False:
            # try to find the title with some things in between
            for year in [item_year, item_year + 1, item_year - 1]:
                for title in title_id_mapping_20ml:
                    m = re.match(f'{item_title}.*\({year}\)', title)
                    if m:
                        # print(f"{title}  --->  {item_ty}")
                        ml20_id = title_id_mapping_20ml[title]
                        if ml20_id in tags_id:
                            item.append(" ".join(tags_id[ml20_id]))
                        found = True
                        break
                if found:
                    break
        if found is False:
            # try find items with +1 or -1 release year
            print(item)
            item.append("")
        writer.writerow(item)
