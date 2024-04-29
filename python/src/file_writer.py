def write_to_file(file_name, data):
    path = "files/results/" + file_name
    with open(path, "a") as f:
        for line in data:
            f.write(f"{line[0]}, {line[1]}, {line[2]}, {line[3]}\n")