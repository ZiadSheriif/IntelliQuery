import numpy as np
from worst_case_implementation import VecDBWorst
from best_case_implementation import VecDBBest

class DataApi:
  def __init__(self, file_path, worst = False, database_path="./DataBase") -> None:
    self.file_path = file_path
    if worst:
      self.db = VecDBWorst(self.file_path,False)
    else:
      self.db = VecDBBest(self.file_path,database_path,False)
    self.chunk_size = 10000

  # Function to generate random embeddings
  def __generate_embeddings(self,num_records, embedding_dim):
      return [np.random.random(embedding_dim).tolist() for _ in range(num_records)]


  def generate_data_file(self,num_of_records):
    # Insert records in chunks
    for i in range(0, num_of_records, self.chunk_size):
        chunk_records = []
        for j in range(i + 1, i + self.chunk_size + 1):
            if j > num_of_records:
                break
            record = {"id": j, "embed": self.__generate_embeddings(1, 70)[0]}
            chunk_records.append(record)

        self.insert_records_binary(chunk_records)
        print(f"Inserted {len(chunk_records)} records. Total records inserted: {j}")

    print("Insertion complete.")


  def insert_records_binary(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
      with open(self.file_path, "ab") as fout:  # Open the file in binary mode for appending
          for row in rows:
              id, embed = row["id"], row["embed"]
              # Pack the data into a binary format
              data = struct.pack(f"I{70}f", id, *embed)
              fout.write(data)
      # self._build_index()
      

  def __generate_embeddings(self,num_records, embedding_dim):
    '''
    Function to generate random embeddings
    '''
    return [np.random.random(embedding_dim).tolist() for _ in range(num_records)]


  def calculate_offset(self, record_id: int) -> int:
      # Calculate the offset for a given record ID
      record_size = struct.calcsize("I70f")
      return (record_id - 1) * record_size


  def get_record_by_id(self, record_id: int) -> Dict[int, Annotated[List[float], 70]]:
      record_size = struct.calcsize("I70f")
      offset = self.calculate_offset(record_id)

      with open(self.file_path, "rb") as fin:
          fin.seek(offset)  # Move the file pointer to the calculated offset
          data = fin.read(record_size)
          if not data:
              return {}  # Record not found

          # Unpack the binary data into a dictionary
          unpacked_data = struct.unpack("I70f", data)
          id_value, floats = unpacked_data[0], unpacked_data[1:]

          # Create and return the record dictionary
          record = {"id": id_value, "embed": list(floats)}
          return {record_id: record}
      

  def get_multiple_records_by_ids(self, records_id: List[int]):
      record_size = struct.calcsize("I70f")
      records = {}

      with open(self.file_path, "rb") as fin:
          for i in range(len(records_id)):
              offset = self.calculate_offset(records_id[i])
              fin.seek(offset)  # Move the file pointer to the calculated offset
              data = fin.read(record_size)
              if not data:
                  records[records_id[i]] = None
                  continue

              # Unpack the binary data into a dictionary
              unpacked_data = struct.unpack("I70f", data)
              id_value, floats = unpacked_data[0], unpacked_data[1:]

              # Create and return the record dictionary
              record = {"id": id_value, "embed": list(floats)}
              records[records_id[i]] = record
      return records
  
  def get_top_k_records(self,k):
      records = []
      record_size = struct.calcsize("I70f")
      with open(self.file_path,'rb') as fin:
          fin.seek(0)
          for i in range(k):
              data = fin.read(record_size)
              unpacked_data = struct.unpack("I70f", data)
              id_value, floats = unpacked_data[0], unpacked_data[1:]

              record = {"id": id_value, "embed": list(floats)} 
              records.append(record)
          return records

  def get_multiple_records_by_ids(self,record_ids):
    return self.db.read_multiple_records_by_id(record_ids)

  def insert_records_binary(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
    return self.db.insert_records_binary(rows)
