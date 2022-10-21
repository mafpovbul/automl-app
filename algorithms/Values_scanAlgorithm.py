"""
Parser for values scan node type
"""

import json
import Annotation

def Values_scanAlgorithm(plan, start=False):
    """ Value Scan Parser """
    result = Annotation.getConjuction(start)
    result += "It will perform a scan over the queried values."

    return result


    
