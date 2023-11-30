from api import DataApi



api_data = DataApi("test.bin")

# api_data.generate_data_file(5000)

# print(api_data.get_record_by_id(5000))

records = api_data.get_multiple_records_by_ids([2, 1, 5])
print(records[5])