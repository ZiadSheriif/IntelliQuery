import numpy as np
from worst_case_implementation import VecDBWorst
from best_case_implementation import VecDBBest
import argparse
from utils import extract_embeds_array
import pandas as pd
from api import DataApi
import os
import time
from dataclasses import dataclass
from typing import List

AVG_OVERX_ROWS = 1

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db1,db2, np_rows, top_k, num_runs):
    results_worst = []
    results_best = []
    for _ in range(num_runs):
        query = np.random.random((1,70))

        # worst
        tic = time.time()
        db_ids_worst = db1.retrive(query,top_k)
        toc = time.time()
        run_time_worst = toc - tic
        
        # best
        tic = time.time()
        db_ids_best = db2.retrive(query,top_k)
        toc = time.time()
        run_time_best = toc - tic
        
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]

        toc = time.time()
        np_run_time = toc - tic
    
        print("=======================================")
        print("Best ids: ",db_ids_best)
        print("Actual ids: ",actual_ids[:top_k])
        print("Worst ids: ",db_ids_worst)
        print("Intersect: ",set(actual_ids[:top_k]).intersection(set(db_ids_best)))
        print("Intersection in top k indices in the best DB: ",find_indices(actual_ids[:top_k], db_ids_best))
        print("=======================================")
        results_worst.append(Result(run_time_worst, top_k, db_ids_worst, actual_ids))
        results_best.append(Result(run_time_best, top_k, db_ids_best, actual_ids))
    return results_worst, results_best

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def find_indices(list1, list2):
    """
    Find the indices of elements of list1 in list2.
    
    :param list1: The list containing elements whose indices are to be found.
    :param list2: The list in which to search for elements from list1.
    :return: A list of indices.
    """
    indices = []
    for element in list1:
        # Convert both to numpy arrays for consistent handling
        np_list2 = np.array(list2)
        # Find the index of element in list2
        found_indices = np.where(np_list2 == element)[0]
        if found_indices.size > 0:
            indices.append(found_indices[0])
    
    return indices

if __name__ == "__main__":
    print("Hello Semantic LSH")
    
    number_of_records = 100000
    number_of_features = 70
    number_of_queries = 5
    top_k = 10
    print("******************************""")
    print("Number of records: ",number_of_records)
    print("Number of queries: ",number_of_queries)
    print("Top k: ",top_k)
    print("******************************""")
    
    
    folder_name = "DataBase"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Mode
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-d','--debug', help='Description of the -d flag', action='store_true')
    args = parser.parse_args()

    # worst_db = VecDBWorst('./DataBase/data.csv',new_db=not args.debug)
    worst_api = DataApi('./DataBase/data_worst.csv',True)
    # best_db = VecDBBest('./DataBase/data.bin','./DataBase',new_db=not args.debug)
    best_api = DataApi('./DataBase/data.bin', False,'./DataBase' )

    if args.debug:
        print("Debug")
        # records_np = pd.read_csv('./DataBase/data.csv',header=None)
        # rows_without_first_element = np.array([row[1:].tolist() for _, row in records_np.iterrows()])
        # records_np=rows_without_first_element
    else:

        # records_database = np.array(best_api.get_first_k_records(10000))
        print("Generating data files")
        records_np = np.random.random((number_of_records, number_of_features))
        # records_np = extract_embeds_array(records_database)

        # records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
        
        records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
        # records_dict = records_database
        _len = len(records_np)

        worst_api.insert_records(records_dict)
        best_api.insert_records_binary(records_dict)

    # query = np.array([best_api.get_multiple_records_by_ids([200])[200]['embed']])
    # print(best_api.get_multiple_records_by_ids([200])[200]['embed'])
    # print(query)

    # res = run_queries(worst_api, records_np, 5, 3)
    # print("Worst:",eval(res))

    # res = run_queries(best_api, records_np, 5, 3)
    # print("Best:",eval(res))
    results_worst, results_best = run_queries(worst_api,best_api, records_np, top_k, number_of_queries)
    print("Worst:",eval(results_worst))
    print("Best:",eval(results_best))

    # records_np = np.concatenate([records_np, np.random.random((90000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # worst_db.insert_records(records_dict)
    # res = run_queries(worst_db, records_np, 5, 10)
    # print(eval(res))

    # records_np = np.concatenate([records_np, np.random.random((900000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # worst_db.insert_records(records_dict)
    # res = run_queries(worst_db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((4000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i +  _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    