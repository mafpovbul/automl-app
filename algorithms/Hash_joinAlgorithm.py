"""
File for Hash Join 
"""

import json
import Annotation

def Hash_joinAlgorithm(plan, start=False):
    """ Hash join parser """
    result = ""

    result += Annotation.parsePlan(plan["Plans"][1], start) + " "
    result += Annotation.parsePlan(plan["Plans"][0]) + " "

    result += "Hash join the result from the earlier operation "
    result += plan["Join Type"] + " Join"
    if 'Hash Cond' in plan:
        result += ' with the condition ' + plan['Hash Cond'].replace("::text", "") + '.'
    else:
        result += '.'

    return result


