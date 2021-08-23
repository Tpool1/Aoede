import os

def clear_user_data():
    root = "data\\profiles"
    profile_list = os.listdir(root)

    for profile in profile_list:
        profile_path = os.path.join(root, profile)
        data_list = os.listdir(profile_path)

        for data in data_list:
            data_path = os.path.join(profile_path, data)
            os.remove(data_path)
            