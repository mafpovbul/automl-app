"""
Parser for sort node type
"""

import json
import Annotation

def SortAlgorithm(plan, start=False):
    """ Sort Parser """
    result = ""

    # Get the text of it's child before if exists
    if "Plans" in plan:
        for child in plan["Plans"]:
            temp = Annotation.parsePlan(child, start)
            result += temp + " "
            if start:
                start = False

    # Parse the Sort
    if plan["Node Type"] == "Sort":
        result += Annotation.getConjuction(start)
        result += "Output is sorted based on attribute "
        if "DESC" in str(plan["Sort Key"]):
            result += str(plan["Sort Key"]).replace('DESC', '')+" in descending order."

        else:
            result += str(plan["Sort Key"])+ "."

    return result



