from pymongo import MongoClient
import json

def get_mongo_obj():
    client = MongoClient()
    return client

def close_mongo_obj(client):
    client.close()

def read_swot():
    client = get_mongo_obj()
    pre_ops = False
    mongo_data = []
    if "swot_analysis" in client.list_database_names():
        mongo_data = list(client.swot_analysis.albert_data.find({},{'_id': 0}))
        mongo_data = []
    if not mongo_data:
        pre_ops = True
        print("Fetch Fresh Data")
        query = [
            {"$unwind":"$company_data"},
            {"$match":{
                "company_data.source_name" : "kable",
                "company_data.swot" : {"$ne": {}}}},
            {"$project": {
                "_id":0,
                "source_name" : "$company_data.source_name",
                "swot_data" : "$company_data.swot",
                "company_name" :1,
                "company_id" :1
            }}]
        mongo_data = client.tata_company_data.swot_collection.aggregate(query)
    close_mongo_obj(client)
    return mongo_data, pre_ops

def format_result(raw_data):
    _picked = []
    swot_data = []
    for e in raw_data:
        for _type, _val in e['swot_data'].items():
            for e_swot in _val:
                if e_swot['Description'] not in _picked:

                    swot_data.append({
                        "description" : e_swot['Description'],
                        "type" : _type,
                        "word_count" : len(e_swot['Description'].split())
                        })
                    _picked.append(e_swot['Description'])
                    if e_swot['summarized'].lower() != e_swot['Description'].lower() \
                            and e_swot['summarized'] not in _picked:
                        swot_data.append({
                            "description" : e_swot['summarized'],
                            "type" : _type,
                            "word_count" : len(e_swot['summarized'].split())
                            })
                        _picked.append(e_swot['summarized'])
    return swot_data

def generate_model_data(swot_data, model='albert'):
    pass

def save_data(swot_data, place="mongo"):
    client = get_mongo_obj()
    # if "swot_analysis" in client.list_database_names():
    with open('data/swot_analysis_data.json', 'w') as f:
        json.dump(swot_data, f)
    client["swot_analysis"]["albert_data"].insert_many(swot_data)
    print("Data Saved in Mongo")
    close_mongo_obj(client)

def main():
    raw_data, pre_ops = read_swot()
    if pre_ops:
        swot_data = format_result(raw_data)
        save_data(swot_data, place='mongo')
    #model_input = generate_model_data(swot_data, model='albert')


if __name__ == '__main__':
    main()

