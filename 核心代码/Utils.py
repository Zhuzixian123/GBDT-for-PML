#用于生成居中字符
def add_spaces(string, num_spaces,type):
    num_spaces1 = num_spaces - len(string)
    num_spaces2 = num_spaces - 2 * len(string)
    spaces1 = ' ' * num_spaces1
    spaces2 = ' ' * num_spaces2
    if type=='英':
        return spaces1 + string + spaces1
    elif type=='中':
        return spaces2 + string + spaces1
