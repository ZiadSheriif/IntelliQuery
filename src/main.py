from api import DataApi


api_data = DataApi("test.bin")

# api_data.generate_data_file(5000)


records = api_data.get_multiple_records_by_ids([2, 1, 5, 8000])
print(records[8000])
