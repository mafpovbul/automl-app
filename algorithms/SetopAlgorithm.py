"""
Parser for SetOp node type
"""

import json
import Annotation

def SetopAlgorithm(plan, start=False):
    """ SetOp Parser """
    result = Annotation.parsePlan(plan["Plans"][0], start)
    result += " " + Annotation.getConjuction()
    result += "Detect "
    cmd_name = str(plan["Command"])
    if cmd_name == "Except" or cmd_name == "Except All":
        result += "differences "
    else:
        result += "similarities "
    result += "between the scanned relations using the "
    result += plan["Node Type"] + " operation."

    return result


