import os
import json
import numpy as np

left_leg_prop_sitting = []
left_leg_ang_sitting = []
right_leg_prop_sitting = []
right_leg_ang_sitting = []
left_leg_prop_standing = []
left_leg_ang_standing = []
right_leg_prop_standing = []
right_leg_ang_standing = []

 
# iterate over all features files
for filename in os.listdir('features'):
    filepath = os.path.join('features', filename)
    # checking if it is a file
    if os.path.isfile(filepath):
        # read file
        file = open(filepath,'r')
        file_content  = file.read()
        json_content = json.loads(file_content)
        file.close()

        # append features
        if filepath.lower().endswith('sitting.json'):
            left_leg_prop_sitting.append(json_content['left_leg']['proportion']['value'])
            left_leg_ang_sitting.append(json_content['left_leg']['angle']['value'])
            right_leg_prop_sitting.append(json_content['right_leg']['proportion']['value'])
            right_leg_ang_sitting.append(json_content['right_leg']['angle']['value'])
        
        if filepath.lower().endswith('standing.json'):
            left_leg_prop_standing.append(json_content['left_leg']['proportion']['value'])
            left_leg_ang_standing.append(json_content['left_leg']['angle']['value'])
            right_leg_prop_standing.append(json_content['right_leg']['proportion']['value'])
            right_leg_ang_standing.append(json_content['right_leg']['angle']['value'])

# print mean and std of features
print(np.mean(left_leg_prop_sitting))
print(np.mean(left_leg_ang_sitting))
print(np.mean(right_leg_prop_sitting))
print(np.mean(right_leg_ang_sitting))
print(np.mean(left_leg_prop_standing))
print(np.mean(left_leg_ang_standing))
print(np.mean(right_leg_prop_standing))
print(np.mean(right_leg_ang_standing))

print(np.std(left_leg_prop_sitting))
print(np.std(left_leg_ang_sitting))
print(np.std(right_leg_prop_sitting))
print(np.std(right_leg_ang_sitting))
print(np.std(left_leg_prop_standing))
print(np.std(left_leg_ang_standing))
print(np.std(right_leg_prop_standing))
print(np.std(right_leg_ang_standing))
