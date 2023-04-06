file_name = 'data/glove.840B.300d.txt'

with open(file_name, "r", encoding="utf8") as f:
    lines = f.readlines()

print(len(lines))


line = lines[0]
# print(line)

print( line.split() )

# print(word, vec)