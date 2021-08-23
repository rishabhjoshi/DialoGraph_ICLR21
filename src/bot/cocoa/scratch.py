import csv
from collections import defaultdict

discourse_words = defaultdict(list)
tsv_filename = 'data/discourse.tsv'
reader = csv.DictReader(open(tsv_filename, 'r'), delimiter='\t')
for row in reader:
	for word_group in reader.fieldnames:  # iterate thru keys (header)
		if row[word_group]:
			discourse_words[word_group].append(row[word_group])

print discourse_words
import pdb; pdb.set_trace()
