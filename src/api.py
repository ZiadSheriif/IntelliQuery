import numpy as np
from worst_case_implementation import VecDBWorst

class DataApi:
  def __init__(self, file_path) -> None:
    self.file_path = file_path
    self.db = VecDBWorst(self.file_path,False)
    self.chunk_size = 10000

  # Function to generate random embeddings
  def __generate_embeddings(self,num_records, embedding_dim):
      return [np.random.rand(embedding_dim).tolist() for _ in range(num_records)]


  def generate_data_file(self,num_of_records):
    # Insert records in chunks
    for i in range(0, num_of_records, self.chunk_size):
        chunk_records = []
        for j in range(i + 1, i + self.chunk_size + 1):
            if j > num_of_records:
                break
            record = {"id": j, "embed": self.__generate_embeddings(1, 70)[0]}
            chunk_records.append(record)

        self.db.insert_records_binary(chunk_records)
        print(f"Inserted {len(chunk_records)} records. Total records inserted: {j}")

    print("Insertion complete.")


  def get_record_by_id(self,record_id):
    return self.db.read_record_by_id(record_id)

  def get_first_k_records(self,k):
    return self.db.get_top_k_records(k)

  def get_multiple_records_by_ids(self,record_ids):
    return self.db.read_multiple_records_by_id(record_ids)