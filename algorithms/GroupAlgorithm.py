"""
File to parse node type group
"""

import json
import Annotation

def GroupAlgorithm(plan, start=False):
    """Parser function for node type limit"""
    parsedPlan = Annotation.parsePlan(plan["Plans"][0], start)
    parsedPlan += " " + Annotation.getConjuction()
    if len(plan["Group Key"]) == 1:
        parsedPlan += "Group the previous operation output with the key "
        parsedPlan += plan["Group Key"][0].replace("::text", "") + "."
    else:
        parsedPlan += "Group the previous operation output with the key "
        for i in plan["Group Key"][:-1]:
            parsedPlan += i.replace("::text", "") + ", "
        parsedPlan += plan["Group Key"][-1].replace("::text", "") + "."

    return parsedPlan



