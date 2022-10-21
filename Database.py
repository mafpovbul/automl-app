import psycopg2
from typing import List, Dict

class Database:

    def __init__(self, db_conf: Dict):

        # Set config
        self.host_name = db_conf["host"]
        self.port_name = db_conf["port"]
        self.db_name = db_conf["database"]
        self.username = db_conf["username"]
        self.password = db_conf["password"]

        # Connect to database
        connectionString = "host='%s' port='%s' dbname='%s' user='%s' password='%s'" % (
            self.host_name, self.port_name, self.db_name, self.username, self.password)
        self.connection = psycopg2.connect(connectionString)
        self.cursor = self.connection.cursor()

    def get_explanations(self, query: str, params_permutation: List[Dict])-> List[Dict]:
        explanations = []
        for param in params_permutation:
            explanations.append(self.explain(query, param))

        return explanations

    def explain(self, query: str, params: Dict)-> Dict:

        # Add the config settings based on params
        res = "BEGIN;"
        for key in params.keys():
            if params[key] == True:
                res += "SET {} TO {};".format(key, "TRUE")
            else:
                res += "SET {} TO {};".format(key, "FALSE")

        # Query the database
        res += "EXPLAIN (FORMAT JSON) " + query
        try:
            self.cursor.execute(res)
            plan = self.cursor.fetchall()
            queryPlan = plan[0][0][0]["Plan"]
        except:
            queryPlan = None

        # Rollback the config changes
        self.cursor.execute("ROLLBACK;")

        return queryPlan


