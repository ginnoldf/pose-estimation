import json
import argparse
import os
import scipy.stats
from math import sqrt, degrees, atan2


# load the keyoints from a json file
def load_pose_keypoints(filepath):   
    # read file
    file = open(filepath,'r')
    file_content  = file.read()
    json_content = json.loads(file_content)
    file.close()

    # read keypoints from json content
    # we assume there is only one persons keypoints in the file and always take the keypoints for the first person
    person = json_content['people'][0]
    pose_keypoints_unformatted = person['pose_keypoints_2d']

    # we want a nicer format
    pose_keypoints = []
    for i in range(0,len(pose_keypoints_unformatted),3):
        keypoint = {
            'x':pose_keypoints_unformatted[i],
            'y':pose_keypoints_unformatted[i+1],
            'c':pose_keypoints_unformatted[i+2]
        }
        pose_keypoints.append(keypoint)

    return pose_keypoints


# calculate the distance between two given keypoints
def distance_kp(a, b):
    # distance between two points in two dimensions
    distance = sqrt((b['x'] - a['x'])**2 + (b['y'] - a['y'])**2)
    
    # multiply the confidences to get a notion of confidence for the distance
    confidence = a['c'] * b['c']

    return distance, confidence


# calculate the angle between three given keypoints, where b is in the middle
def angle_kp(a, b, c):
    # calculate the angle in degrees and return in [0, 360]
    # see https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    angle = degrees(atan2(c['y']-b['y'], c['x']-b['x']) - atan2(a['y']-b['y'], a['x']-b['x']))
    if angle < 0:
        angle += 360
    
    # multiply the confidences to get a notion of confidence for the angle
    confidence = a['c'] * b['c'] * c['c']
    return angle, confidence


# calculate the features for one limb
def features_limb(upper, middle, lower):
    upp_len, upp_conf = distance_kp(upper, middle)
    low_len, low_conf = distance_kp(middle, lower)
    prop = upp_len / low_len
    prop_conf = upp_conf * low_conf
    angle, angle_conf = angle_kp(upper, middle, lower)

    features = {
        'upper_len': {
            'value': upp_len,
            'confidence': upp_conf,
        },
        'lower_len': {
            'value':low_len,
            'confidence':low_conf,
        },
        'proportion': {
            'value': prop,
            'confidence':prop_conf,
        },
        'angle': {
            'value': angle,
            'confidence':angle_conf,
        },
    }

    return features


# get all desired features from the pose keypoints
def calculate_features(kps):
    features = {
        'left_leg': features_limb(kps[12], kps[13], kps[14]),
        'right_leg': features_limb(kps[9], kps[10], kps[11]),
        'left_arm': features_limb(kps[5], kps[6], kps[7]),
        'right_arm': features_limb(kps[2], kps[3], kps[4]),
    }

    return features


# estimate the pose by calculating probabilities for the features assuming the two poses
def estimate_pose(features):
    # define average and standard deviation for gaussian distributions
    # we took mean and standard devition from our images
    left_leg_proportion_standing_avg = 1.2
    left_leg_proportion_standing_std = 0.1
    left_leg_proportion_sitting_avg = 0.4
    left_leg_proportion_sitting_std = 0.1

    left_leg_angle_standing_avg = 180
    left_leg_angle_standing_std = 10
    left_leg_angle_sitting_avg = 250
    left_leg_angle_sitting_std = 10

    right_leg_proportion_standing_avg = 1.2
    right_leg_proportion_standing_std = 0.1
    right_leg_proportion_sitting_avg = 0.4
    right_leg_proportion_sitting_std = 0.1

    right_leg_angle_standing_avg = 180
    right_leg_angle_standing_std = 10
    right_leg_angle_sitting_avg = 110
    right_leg_angle_sitting_std = 10

    # define gaussian distributions for the features
    left_leg_proportion_standing_dist = scipy.stats.norm(left_leg_proportion_standing_avg, left_leg_proportion_standing_std)
    left_leg_proportion_sitting_dist = scipy.stats.norm(left_leg_proportion_sitting_avg, left_leg_proportion_sitting_std)
    left_leg_angle_standing_dist = scipy.stats.norm(left_leg_angle_standing_avg, left_leg_angle_standing_std)
    left_leg_angle_sitting_dist = scipy.stats.norm(left_leg_angle_sitting_avg, left_leg_angle_sitting_std)

    right_leg_proportion_standing_dist = scipy.stats.norm(right_leg_proportion_standing_avg, right_leg_proportion_standing_std)
    right_leg_proportion_sitting_dist = scipy.stats.norm(right_leg_proportion_sitting_avg, right_leg_proportion_sitting_std)
    right_leg_angle_standing_dist = scipy.stats.norm(right_leg_angle_standing_avg, right_leg_angle_standing_std)
    right_leg_angle_sitting_dist = scipy.stats.norm(right_leg_angle_sitting_avg, right_leg_angle_sitting_std)

    # get probabilities of the given pose, assuming the person is standing
    # by multiplying the probability with the confidence, parts with a small confidence will have a small impact on the result
    left_leg_proportion_standing_prob = left_leg_proportion_standing_dist.pdf(features['left_leg']['proportion']['value']) * features['left_leg']['proportion']['confidence']
    left_leg_angle_standing_prob = left_leg_angle_standing_dist.pdf(features['left_leg']['angle']['value']) * features['left_leg']['angle']['confidence']
    right_leg_proportion_standing_prob = right_leg_proportion_standing_dist.pdf(features['right_leg']['proportion']['value']) * features['right_leg']['proportion']['confidence']
    right_leg_angle_standing_prob = right_leg_angle_standing_dist.pdf(features['right_leg']['angle']['value']) * features['right_leg']['angle']['confidence']
    
    # get probabilities of the given pose, assuming the person is sitting
    left_leg_proportion_sitting_prob = left_leg_proportion_sitting_dist.pdf(features['left_leg']['proportion']['value']) * features['left_leg']['proportion']['confidence']
    left_leg_angle_sitting_prob = left_leg_angle_sitting_dist.pdf(features['left_leg']['angle']['value']) * features['left_leg']['angle']['confidence']
    right_leg_proportion_sitting_prob = right_leg_proportion_sitting_dist.pdf(features['right_leg']['proportion']['value']) * features['right_leg']['proportion']['confidence']
    right_leg_angle_sitting_prob = right_leg_angle_sitting_dist.pdf(features['right_leg']['angle']['value']) * features['right_leg']['angle']['confidence']

    # average over the pose probabilities to get a final probability
    standing_prob = (left_leg_proportion_standing_prob + left_leg_angle_standing_prob + right_leg_proportion_standing_prob + right_leg_angle_standing_prob)/4
    sitting_prob = (left_leg_proportion_sitting_prob + left_leg_angle_sitting_prob + right_leg_proportion_sitting_prob + right_leg_angle_sitting_prob)/4

    print('P(keypoints|standing) = ' + str(standing_prob))
    print('P(keypoints|sitting) = ' + str(sitting_prob))

    # return the pose with the higher probability
    return 'sitting' if sitting_prob > standing_prob else 'standing'


def main():
    # pass the filepath as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help='path to the json file containing the pose keypoints')
    args = parser.parse_args()

    # read pose keypoint json path argument
    filepath = args.path
    filename = os.path.basename(filepath)

    # load pose keypoints from json
    pose_keypoints = load_pose_keypoints(filepath)

    # calculate the features from the pose keypoints
    features = calculate_features(pose_keypoints)
    
    # save the features
    with open(os.path.join('features', filename), 'w') as outfile:
        json.dump(features, outfile, indent=4)

    # finally estimate the pose and print the result
    pose = estimate_pose(features)
    print('The estimated pose for the given keypoints is ' + pose)


if __name__ == '__main__':
    main()
