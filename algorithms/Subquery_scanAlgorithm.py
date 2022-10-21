"""
Subquery scan parser
"""

import json
import Annotation

def Subquery_scanAlgorithm(plan, start=False):
    """ Subquery scan parser """
    result = ''

    if 'Plans' in plan:
        for child in plan['Plans']:
            result += Annotation.parsePlan(child, start) + " "
            if start:
                start = False

    result += Annotation.getConjuction(start)
    result += 'Perform subquery scan on the output of '
    result += 'earlier operations without any changes to the result '
    return result


