list = []
with open('train.txt', 'r') as f:
    for line in f:
        list.append(line.strip())

with open("train.txt", "w") as f:
    for item in sorted(list):
        f.writelines(item)
        f.writelines('\n')
    f.close()
