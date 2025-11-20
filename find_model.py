import os

print("\n Searching for model.pkl...\n")

for root, dirs, files in os.walk(".", topdown=True):
    for file in files:
        if file == "model.pkl":
            print("FOUND model.pkl HERE:")
            print(os.path.abspath(os.path.join(root, file)))
            print()
