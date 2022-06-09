import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="visualize output with all templates result json")
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="path of the result json",
        required=True,
    )
    parser.add_argument(
        "--score_only",
        action="store_true",
        help=(
            "If passed, only output score column."
        ),
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    result_json_path = args.input_json
    results = json.load(open(result_json_path))
    output_path = result_json_path[:-5] + ".txt"
    print(results[0]['dataset_name'],results[0]['dataset_config_name'])
    lines = []
    for item in results:
        if args.score_only:
            line = str(item['evaluation']['accuracy'])
        else:
            line = f"{item['template_name']}\t{item['evaluation']['accuracy']}"
        lines.append(line)
        print(line)
    
    with open(output_path, 'w') as out:
        for line in lines:
            out.write(line)
            out.write('\n')