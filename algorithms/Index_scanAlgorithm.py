"""
Parser for index scan node type
"""

import json
import Annotation

def Index_scanAlgorithm(plan, start=False):
    """ Index Scan parser """
    result = Annotation.getConjuction(start)

    #Parse the index scan or index only scan
    if plan["Node Type"] == "Index Scan":
        result += "Perform index scan with index table "+ plan["Index Name"]
        if "Index Cond" in plan:
            result += " based on the conditions "+ plan["Index Cond"].replace('::text', '')
        result += ". Once done, matched rows in " + plan["Relation Name"]
        result += " table are returned."

        if "Filter" in plan:
            result += " Filter output by "+ plan["Filter"].replace('::text', '') +"."

    elif plan["Node Type"] == "Index Only Scan":
        result += "Perform index scan with index table "+ plan["Index Name"]
        if "Index Cond" in plan:
            result += " with condition(s) "+ plan["Index Cond"].replace('::text', '')
        result += ". Return any matches found as the result."
        if "Filter" in plan:
            result += " Filter output by "+ plan["Filter"].replace('::text', '') +"."

    return result


