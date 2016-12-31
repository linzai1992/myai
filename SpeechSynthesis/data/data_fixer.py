allowed_symbols = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", ",", "'", " "]

def line_is_problem(line):
    for c in line:
        if not c in allowed_symbols:
            return True
    return False

def format_line(line):
    tokens = line.split(" ")
    new_line = tokens[0] + "-"
    for i in range(1, len(tokens)):
        if tokens[i].strip() != "":
            new_line += tokens[i] + "_"
    return new_line[:-1]

def fix_data(file_path, output_path):
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        with open(output_path, "w", encoding="utf-8") as out:
            line_index = 0
            for line in f:
                line_index += 1
                lower_line = line.lower().strip()
                if not line_is_problem(lower_line):
                    out.write(format_line(lower_line) + "\n")
                    if line_index % 10000 == 0:
                        print("Processed %i" % (line_index))

fix_data("cmudict_0_7b.txt", "cmudict_proc.txt")
