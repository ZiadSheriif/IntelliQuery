import struct

# Define the binary file name
binary_file_name = 'records_with_index_name.bin'

# Define the index of the element you want to access
i = 9999  # Change this to the desired index

# Calculate the position of the ith element based on record size
record_size = struct.calcsize('I20s20s')  # Size of packed data
print(record_size)
position = i * record_size

# Get the address of the first block in the binary file
binary_file_address = 0
# with open(binary_file_name, 'rb') as file:
#     binary_file_address = file.tell()
#     print(binary_file_address)

# Calculate the absolute position of the ith element
absolute_position = binary_file_address + position

# Open the binary file and seek to the absolute position of the ith element
with open(binary_file_name, 'rb') as file:
    file.seek(absolute_position)

    # Read the packed data at the ith position
    packed_data = file.read(record_size)

    # Unpack the data
    index, name, phone = struct.unpack('I20s20s', packed_data)
    name = name.decode().strip('\0')
    phone = phone.decode().strip('\0')

    print(f'Index: {index}, Name: {name}, Phone: {phone}')
