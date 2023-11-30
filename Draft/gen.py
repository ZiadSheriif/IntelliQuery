from worst_case_implementation import VecDBWorst
import numpy as np

# Function to generate random embeddings
def generate_embeddings(num_records, embedding_dim):
    return [np.random.rand(embedding_dim).tolist() for _ in range(num_records)]

# Create an instance of VecDB
db = VecDBWorst()

# Define parameters
total_records = 10000  # 20 million records
chunk_size = 10000  # Insert records in chunks of 10,000

# Insert records in chunks
for i in range(0, total_records, chunk_size):
    chunk_records = []
    for j in range(i + 1, i + chunk_size + 1):
        if j > total_records:
            break
        record = {"id": j, "embed": generate_embeddings(1, 70)[0]}
        #  make this size of record to be fixed 1500 bytes
        # size_of_dummy_needed = 1500 - len(record["embed"])
        
        chunk_records.append(record)

    db.insert_records(chunk_records)
    print(f"Inserted {len(chunk_records)} records. Total records inserted: {j}")

print("Insertion complete.")
