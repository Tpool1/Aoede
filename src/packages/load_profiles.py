import os
from packages.profile import profile

def load_profiles():
    root = "data\\profiles"
    names = os.listdir(root)

    # list to hold all profile dictionaries
    profiles = []

    for name in names:

        # dictionary for each profile
        profile_dict = {}

        profile_path = os.path.join(root, name)
        
        info_path = os.path.join(profile_path, "info.txt")

        with open(info_path, 'r') as f:
            # get list of lines in info.txt
            profile_info = f.readlines()

        for info in profile_info:
            info = info.split()

            # set i to one to get the element in front of value in list
            i = 1
            for value in info:
                if value == 'Name:':
                    name = info[i]
                    profile_dict["Name"] = name

                i = i + 1

        profiles.append(profile_dict)

    i = 0 
    for data in profiles:
        name = data['Name']
        p = profile(name)

        profiles[i] = p

        i = i + 1

    return profiles
    