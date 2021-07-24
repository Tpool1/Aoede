def remove_breaks(self, val_list):
    i = 0
    for var in val_list: 
        new_var = var.replace('\n','')
        val_list[i] = new_var
        i = i + 1 
    
    return val_list
    