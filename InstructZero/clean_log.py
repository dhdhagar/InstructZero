file_name = "logfile_wizardlm-vector-similarity-kernel.log"

with open(file_name, "r") as ff:
    lines = ff.readlines()
    new_lines = []
    for line in lines:
        if "message='OpenAI API response'" in line:
            continue
        new_lines.append(line)

# store in new log
with open(file_name, "w") as ff:
    for line in new_lines:
        ff.write(line)
