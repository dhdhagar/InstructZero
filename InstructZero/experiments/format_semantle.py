import json
import csv
import random


def format_semantle(file_name):
    # Load the csv data
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    num_examples = len(data) - 1
    examples = []
    for i in range(1, num_examples):
        guess_word, similarity_score = data[i]
        examples.append({"input": guess_word, "output": similarity_score})
    random.shuffle(examples)

    eval_size = 100
    train_size = num_examples - eval_size

    train_examples = examples[:train_size]
    eval_examples = examples[train_size:]

    # Renumber IDs for the training set
    train_dict = {i + 1: train_examples[i] for i in range(len(train_examples))}
    eval_dict = {i + 1: eval_examples[i] for i in range(len(eval_examples))}

    # Create metadata
    train_metadata = {"num_examples": len(train_dict)}
    eval_metadata = {"num_examples": len(eval_dict)}

    # Save the train set
    train_data = {"metadata": train_metadata, "examples": train_dict}
    train_file = file_name.replace("semantle/", "").replace(".csv", ".json")
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)

    # Save the eval set
    eval_data = {"metadata": eval_metadata, "examples": eval_dict}
    eval_file = file_name.replace("induce/semantle/", "execute/").replace(".csv", ".json")
    with open(eval_file, "w") as f:
        json.dump(eval_data, f, indent=4)

    print(f"Data successfully split into {train_file} and {eval_file}")


if __name__ == "__main__":
    file_path = "/scratch/workspace/vpimpalkhute_umass_edu-bo_llm/InstructZero/InstructZero/experiments/data/instruction_induction/raw/induce/semantle/"
    word_list = [
        "birthstone",
        "cement",
        "child",
        "crane",
        "computer",
        "meatloaf",
        "papa",
        "polyethylene",
        "sax",
        "trees",
    ]

    word = word_list[0]
    format_semantle(f"{file_path}{word}.csv")
