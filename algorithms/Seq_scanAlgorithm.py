"""
File to parse node type seq scan
"""

import json
import Annotation

def Seq_scanAlgorithm(plan, start=False):
    """ Parser for the Seq Scan Node Type"""
    line = Annotation.getConjuction(start)
    line += "Perform sequential scan on relation "
    if "Relation Name" in plan:
        line += plan['Relation Name']
    if "Alias" in plan:
        if plan['Relation Name'] != plan['Alias']:
            line += " whose alias is "
            line += plan['Alias']
    line += "."
    if "Filter" in plan:
        line += " The condition "
        line += plan['Filter'].replace("::text", "")
        line += " binds this operation"
        line += "."

    return line


