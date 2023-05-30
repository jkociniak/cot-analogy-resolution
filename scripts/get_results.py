import pickle
import argparse
import os 

argParser = argparse.ArgumentParser()

argParser.add_argument("--file") 
args = argParser.parse_args() 


# Specify the path to your pickle file
pickle_file_path = os.path.join("Results/", args.file)

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    results = pickle.load(file)

#print(content[0])
#print(len(content.keys()))

total = len(results)
correct = sum(1 for i in results if results[i]["correct"])

accuracy = correct / total

print(f"Accuracy: {accuracy * 100}%")
