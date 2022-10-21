"""
Main file to parse query plan
"""

import random

import algorithms.GenericAlgorithm as generic
import algorithms.Hash_joinAlgorithm as hash_join
import algorithms.SortAlgorithm as sort
import algorithms.AggregateAlgorithm as aggregate
import algorithms.Seq_scanAlgorithm as seq_scan
import algorithms.HashAlgorithm as hashAlgorithm
import algorithms.Merge_joinAlgorithm as merge_join
import algorithms.LimitAlgorithm as limit
import algorithms.UniqueAlgorithm as unique
import algorithms.Function_scanAlgorithm as function_scan
import algorithms.Index_scanAlgorithm as index_scan
import algorithms.Values_scanAlgorithm as values_scan
import algorithms.Nested_loopAlgorithm as nested_loop
import algorithms.Cte_scanAlgorithm as cte_scan
import algorithms.AppendAlgorithm as append
import algorithms.MaterializeAlgorithm as materialize
import algorithms.Subquery_scanAlgorithm as subquery_scan
import algorithms.SetopAlgorithm as setop
import algorithms.GroupAlgorithm as group

class selectParser:
    """ selectParserClass """
    def __init__(self):
        """ Init Class """

        self.GenericAlgorithm = generic.GenericAlgorithm

        self.Hash_Join = hash_join.Hash_joinAlgorithm
        self.Sort = sort.SortAlgorithm
        self.Aggregate = aggregate.AggregateAlgorithm
        self.Seq_Scan = seq_scan.Seq_scanAlgorithm
        self.Hash = hashAlgorithm.HashAlgorithm
        self.Merge_Join = merge_join.Merge_joinAlgorithm
        self.Limit = limit.LimitAlgorithm
        self.Unique = unique.UniqueAlgorithm
        self.Function_Scan = function_scan.Function_scanAlgorithm
        self.Index_Scan = index_scan.Index_scanAlgorithm
        self.Index_Only_Scan = index_scan.Index_scanAlgorithm
        self.Values_Scan = values_scan.Values_scanAlgorithm
        self.Nested_Loop = nested_loop.Nested_loopAlgorithm
        self.CTE_Scan = cte_scan.Cte_scanAlgorithm
        self.Append = append.AppendAlgorithm
        self.Materialize = materialize.MaterializeAlgorithm
        self.Subquery_Scan = subquery_scan.Subquery_scanAlgorithm
        self.SetOp = setop.SetopAlgorithm
        self.Group = group.GroupAlgorithm

def parsePlan(plan, start=False):

    """ Parse json format of query plan """

    selector = selectParser()
    try:
        parser = getattr(selector, plan["Node Type"].replace(" ", "_"))
    except:
        parser = selector.GenericAlgorithm
    parsedPlan = initPlan(plan, start)
    parsedPlan += parser(plan, start)
    return parsedPlan

LIST_OF_CONJUNCTION = ["Thereafter, ", "Then, ", "Next, ", "Subsequently, ", "Accordingly, "]

def getConjuction(start=False):

    """ Get random conjuction """

    if start:
        return "First of all, "
    return random.choice(LIST_OF_CONJUNCTION)

def initPlan(plan, start=False):

    """ Check for InitPlan """

    result = ""

    if "Parent Relationship" in plan:
        if plan["Parent Relationship"] == "InitPlan":
            result = getConjuction(start)
            result += "The " + plan["Node Type"]
            result += " node and its subsequent child node will be executed first"
            result += " since the result from this node needs to be calculated initially"
            result += " and it is only calculated once for the whole query. "
            result += "The plan is as follows: "

    return result
