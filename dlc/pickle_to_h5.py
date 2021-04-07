"""
pickle_to_h5.py

convert any DeepLabcut .pickle outputs to .h5 files

Sept. 29, 2020
"""
os.environ["DLClight"] = "True"
import deeplabcut
import argparse

# glob for subdirectories
def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
parser.add_argument('-c', '--dlc_config',help='DeepLabCut config .yaml path', default='/home/seuss/Desktop/MathisNetwork2/config.yaml')
parser.add_argument('-d', '--data_path', help='parent directory of pickle files', default='/home/seuss/Desktop/Phils_app/')
args = parser.parse_args()

# deeplabcut.convert_detections2tracklets(args.dlc_config, [args.data_path], videotype='avi')
pickle_list = find('*TOP*bx.pickle', args.data_path)

for vid in pickle_list:
    print('converting to pickle video at ' + vid)
    deeplabcut.convert_raw_tracks_to_h5(args.dlc_config, vid)
print('done converting ' + len(pickle_list) + ' pickles')