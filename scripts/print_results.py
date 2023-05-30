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



def print_dict_with_utf8(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            value = {k: v.encode('utf-8') if isinstance(v, str) else v for k, v in value.items()}
        print(key, value)



# Print the content
print_dict_with_utf8(results)
