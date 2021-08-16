
def write_conversation_data(text):
    f = open("data\\user_data\\conversations.txt", "r")
    lines = f.readlines()
    lines.append(text + "\n")

    f = open("data\\user_data\\conversations.txt", "w")
    f.writelines(lines)
    f.close()
