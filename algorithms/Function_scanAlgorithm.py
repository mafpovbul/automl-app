"""
File to parse node type function scan
"""

import json
import Annotation

def Function_scanAlgorithm(plan, start=False):
    """ Parser for Function Scan node type """
    parsedPlan = Annotation.getConjuction(start)
    parsedPlan += "Executes function " + plan["Function Name"]
    parsedPlan += " and returns the recordset created."
    return parsedPlan


