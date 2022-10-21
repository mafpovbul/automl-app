"""
File to parse node type unique
"""

import json
import Annotation

def UniqueAlgorithm(plan, start=False):
    """Parser function for node type unique"""
    parsedPlan = Annotation.parsePlan(plan["Plans"][0], start) + " "
    parsedPlan += Annotation.getConjuction()
    parsedPlan += "it will scan each sorted data row and"
    parsedPlan += "discard rows with values mirroring the previous row"
    return parsedPlan


