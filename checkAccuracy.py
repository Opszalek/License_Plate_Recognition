import argparse
import json
from pathlib import Path

def check_accuracy():
    parser = argparse.ArgumentParser()
    parser.add_argument('corect_results', type=str)
    parser.add_argument('result', type=str)
    args = parser.parse_args()

    corect_results = Path(args.corect_results)
    result = Path(args.result)
    corect_results = open(corect_results, 'r')
    data = json.load(corect_results)
    results = open(result, 'r')
    results = json.load(results)
    correct = 0
    all_letters = 0
    for key, value in data.items():
        i = 0
        for char in value:
            if len(results[key])>i:
                if char == results[key][i]:
                    correct += 1
            all_letters += 1
            i+=1
    print("Accuracy:",correct/all_letters*100,"%")
def main():

    check_accuracy()



if __name__ == '__main__':
    main()
