
import json

def check_accuracy():
    file_path = 'corect_results.json'
    file = open(file_path, 'r')
    data = json.load(file)
    file_path = 'result.json'
    file = open(file_path, 'r')
    results = json.load(file)
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