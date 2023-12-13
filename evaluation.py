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

import pickle

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

# def run_queries(db1,db2, np_rows, top_k, num_runs,delete=False):
def run_queries(db, np_rows, top_k, num_runs, delete=False):
    results = []
    for i in range(num_runs):
        if delete:
            query = np.random.random((1,70))
            np.save( "./DataBase/q"+str(i)+'.npy',query)
        else:
            query = np.load( "./DataBase/q"+str(i)+'.npy')

        tic = time.time()
        db_ids = db.retrive(query,top_k)
        toc = time.time()
        run_time= toc - tic
              
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis=1).squeeze().tolist()[::-1]

        toc = time.time()
        np_run_time = toc - tic
    
        results.append(Result(run_time,top_k,db_ids,actual_ids))
    return results

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            print('retrieving number not equal to top_k')
            continue

        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    # print("not in top top_k*3")
                    score -= ind
            except:
                # print("not in ids")
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

    
def compare_results_print(worst_res,best_res,top_k):
        for i in range(len(worst_res)):
            actual_ids=worst_res[i].actual_ids
            db_ids_best=best_res[i].db_ids
            db_ids_worst=worst_res[i].db_ids

            run_time_worst=worst_res[i].run_time
            run_time_best=best_res[i].run_time


            print("=======================================")
            print("Best ids: ",db_ids_best)
            print("Actual ids: ",actual_ids[:top_k])
            print("Worst ids: ",db_ids_worst)
            print("Intersect: ",set(actual_ids[:top_k]).intersection(set(db_ids_best)))
            print("Intersection in top k indices in the best DB: ",find_indices(actual_ids[:top_k], db_ids_best))
            
            print("Time taken by Query (Best): ",run_time_best)
            print("Time taken by Query (Worst): ",run_time_worst)
            print("=======================================")
    
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
    parser.add_argument('-d','--delete', help='Description of the -d flag', action='store_true')
    parser.add_argument('-n', '--numberofrecords', type=int, help='Description of the -n flag')
    args = parser.parse_args()

    best_api = DataApi(file_path='./DataBase/data.bin',worst= False,database_path='./DataBase',delete_db=args.delete)
    worst_api = DataApi(file_path='./DataBase/data_worst.csv',worst=True,database_path='./DataBase',delete_db=args.delete)

    if args.numberofrecords is not None:
        print("new number of records: ",args.numberofrecords)
        number_of_records = number_of_records

    if not args.delete:
        print("Reading")
        with open('best_api.pkl', 'rb') as file:
            best_api = pickle.load(file)
        records_database = np.array(best_api.get_first_k_records(number_of_records))
        records_np = extract_embeds_array(records_database)
        records_dict = records_database
        _len = len(records_np)
    else:
        print("Generating data files ........")
        records_np = np.random.random((number_of_records, number_of_features))

        records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
        _len = len(records_np)

        worst_api.insert_records(records_dict)
        best_api.insert_records_binary(records_dict)

        # Save the Object to be read Again
        with open('best_api.pkl', 'wb') as file:
            pickle.dump(best_api, file)
        

 
    # Worst
    res_worst = run_queries(worst_api, records_np, top_k, number_of_queries,args.delete)
    # Best
    res_best = run_queries(best_api, records_np, top_k, number_of_queries,False)

    compare_results_print(res_worst,res_best,top_k)
    print("Worst:",eval(res_worst))
    print("Best:",eval(res_best))

    # res = run_queries(best_api, records_np, 5, 3)
    # print("Best:",eval(res))
    # results_worst, results_best = run_queries(worst_api,best_api, records_np, top_k, number_of_queries)
    # print("Worst:",eval(results_worst))
    # print("Best:",eval(results_best))

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