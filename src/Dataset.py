import ms3
import csv
import torch

table:list[str] = []
with open('beethoven_piano_sonatas\chords\\01-1.chords.tsv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        table.append(row)

output_file = 'tab_str.tsv'

