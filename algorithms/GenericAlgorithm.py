"""
Generic Parser
"""

import json
import Annotation

def GenericAlgorithm(plan, start=False):
    """ Parse unknown node_type """
    parsedPlan = Annotation.getConjuction(start)
    parsedPlan += "Perform " + plan["Node Type"] + "."
    if "Plans" in plan:
        for child in plan["Plans"]:
            parsedPlan += " " + Annotation.parsePlan(child)
    return parsedPlan


