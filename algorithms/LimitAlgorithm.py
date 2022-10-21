"""
File to parse node type limit
"""

import json
import Annotation

def LimitAlgorithm(plan, start=False):
    """Parser function for node type limit"""
    parsedPlan = Annotation.parsePlan(plan["Plans"][0], start)
    parsedPlan += " Scan scope is limited to "
    total_rows = plan["Plan Rows"]
    parsedPlan += str(total_rows) + " entries."
    return parsedPlan

