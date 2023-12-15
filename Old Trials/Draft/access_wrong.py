# Open a file: file
file = open('records_with_index_name_phone.bin',mode='r')

# read all lines at once
all_of_it = file.read()

# close the file
file.close()