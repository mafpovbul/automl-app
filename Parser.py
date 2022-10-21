from typing import List, Dict

class Parser:

    def annotate_plan(self, query_plan: Dict)-> str:

        # Parse plan

        ### ADD CODE HERE ###

        parsed_plan = "parsed " + query_plan # dummy

        return parsed_plan

    def generate_query_costs(self, query_plans: List[Dict])-> List[float]:

        query_costs = []
        for plan in query_plans:
            cost = self.get_query_cost(plan)
            query_costs.append(cost)

        return query_costs

    def get_query_cost(self, query_plan: Dict):

        ### ADD CODE HERE ###

        return 0 #dummy