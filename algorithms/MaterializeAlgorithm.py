"""
Parser for materialize node type
"""

import json
import Annotation

def MaterializeAlgorithm(plan, start=False):
    """ Materialize parser """
    result = ""

    # Get the text of it's child before if exists
    if "Plans" in plan:
        for child in plan["Plans"]:
            temp = Annotation.parsePlan(child, start)
            result += temp + " "
            if start:
                start = False

    #Parse the materialize
    if plan["Node Type"] == "Materialize":
        result += Annotation.getConjuction(start)
        result += "Store the results in memory to eliminate latency and disk storage overhead. "

    return result

