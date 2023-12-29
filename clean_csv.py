import os
import csv


wav_files = set(f[:-4] for f in os.listdir('wavs') if f.endswith('.wav'))
with open('transcriptions.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [row for row in reader if row[0] in wav_files]

with open('transcriptions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)