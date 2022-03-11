# Pose Estimation

This project is the solution of assignment 2 of the NUS module EE4705 Human-Robot Interaction.

This project estimates the pose of a person from OpenPose pose keypoints. We only distinguish between the standing and sitting pose. 
After creating the OpenPose json file, you can execute our script with the following command, in the root folder of this project:

```
python3 main.py --path /path/to/your/keypoints.json
```
In the command line you can see the calculated probabilities for those keypoints assuming the standing or the sitting pose. You can see which pose is estimated. In the features image you will find a json file with the same name as your json file containing the keypoints, which contains our features for your image with confidences.