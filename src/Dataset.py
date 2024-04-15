import ms3

folder_path = "beethoven_piano_sonatas/"

harmonies = ms3.load_tsv(folder_path + 'harmonies/01-1.harmonies.tsv')
notes = ms3.load_tsv(folder_path + 'notes/01-1.notes.tsv')
chords = ms3.load_tsv(folder_path + 'chords/01-1.chords.tsv')
measures = ms3.load_tsv(folder_path + 'measures/01-1.measures.tsv')

print(notes)