"""
Merge Join parser
"""

import json
import Annotation

def Merge_joinAlgorithm(plan, start=False):
    """ Merge Join parser """
    result = ''

    if 'Plans' in plan:
        for child in plan['Plans']:
            result += Annotation.parsePlan(child, start) + " "
            if start:
                start = False

    result += Annotation.getConjuction(start)
    result += 'Merge Join is performed on the output'

    if 'Merge Cond' in plan:
        result += ' with condition ' + plan['Merge Cond'].replace("::text", "")

    if 'Join Type' == 'Semi':
        result += ' however only the left relation\'s row is returned'
    else:
        result += '.'

    return result


