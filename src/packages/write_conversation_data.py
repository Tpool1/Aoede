
def write_conversation_data(text):
    f = open("data\\conversations.txt", "r")
    lines = f.readlines()
    lines.append(text + "\n")

    f = open("data\\conversations.txt", "w")
    f.writelines(lines)
    f.close()
