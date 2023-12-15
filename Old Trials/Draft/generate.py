import struct

# Define the binary file name
binary_file_name = 'records_with_index_name.bin'

# Generate and write the records to the binary file
with open(binary_file_name, 'wb') as file:
    for i in range(10000):
        # Generate example name and phone number (you can replace with your data source)
        name = f"Name-{i}"
        phone = f"Phone-{i}"

        # Ensure a fixed length for name and phone
        name = name.ljust(20, '\0')  # 20 characters
        phone = phone.ljust(20, '\0')  # 20 characters

        # Pack data into binary format (4 bytes for index, 20 bytes for name, and 20 bytes for phone)
        packed_data = struct.pack('I20s20s', i, name.encode(), phone.encode())

        # Write the packed data to the binary file
        file.write(packed_data)