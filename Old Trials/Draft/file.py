import csv

# Generate 10,000 float values based on their index
float_data = [float(i) for i in range(10000)]

# Define the CSV file name
csv_file_name = 'float_records_with_index.csv'

# Write the float data with index to the CSV file
with open(csv_file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Value'])  # Write a header row
    for index, value in enumerate(float_data):
        writer.writerow([index, value])

# Get the address of the first block in the CSV file
csv_file_address = None
with open(csv_file_name, 'rb') as file:
    csv_file_address = file.tell()

print(f'CSV file created: {csv_file_name}')
print(f'Address of the first block in the CSV file: {csv_file_address}')
