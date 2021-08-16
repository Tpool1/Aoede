
def clear_user_data():
    f = open("data\\user_data\\conversations.txt", "r+")
    f.truncate(0)
    f.close()
    