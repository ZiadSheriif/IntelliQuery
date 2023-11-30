from typing import Dict, List, Annotated
import struct
import numpy as np

class VecDBWorst:
    def __init__(self, file_path = "saved_db.bin", new_db = True) -> None:
        self.file_path = file_path
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([float(e) for e in embed])
                fout.write(f"{row_str}\n")
        self._build_index()

    def insert_records_binary(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.file_path, "ab") as fout:  # Open the file in binary mode for appending
            for row in rows:
                id, embed = row["id"], row["embed"]
                # Pack the data into a binary format
                data = struct.pack(f"I{70}f", id, *embed)
                fout.write(data)
        self._build_index()

    def calculate_offset(self, record_id: int) -> int:
        # Calculate the offset for a given record ID
        record_size = struct.calcsize("I70f")
        return (record_id - 1) * record_size

    def read_record_by_id(self, record_id: int) -> Dict[int, Annotated[List[float], 70]]:
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

    def read_multiple_records_by_id(self, records_id: List[int]):
        record_size = struct.calcsize("I70f")
        records = {}

        with open(self.file_path, "rb") as fin:
            for i in range(len(records_id)):
                offset = self.calculate_offset(records_id[i])
                fin.seek(offset)  # Move the file pointer to the calculated offset
                data = fin.read(record_size)
                if not data:
                    records[records_id[i]] = None

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

    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        scores = []
        with open(self.file_path, "r") as fin:
            for row in fin.readlines():
                row_splits = row.split(",")
                id = int(row_splits[0])
                embed = [float(e) for e in row_splits[1:]]
                score = self._cal_score(query, embed)
                scores.append((score, id))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        pass


