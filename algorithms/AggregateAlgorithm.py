"""
Parser for GroupAggregate node type
"""
import Annotation
import json
def AggregateAlgorithm(queryplan, isStart=False):
    """ Parser for Aggregate node type """

    if queryplan["Strategy"] == "Sorted":
        parsedPlan = Annotation.parsePlan(queryplan["Plans"][0], isStart) + " "
        parsedPlan += Annotation.getConjuction()

        if "Group Key" in queryplan:
            parsedPlan += "Group result by "
            for groupKey in queryplan["Group Key"]:
                parsedPlan += groupKey.replace("::text", "") + ", "
            parsedPlan = parsedPlan[:-2]
        if "Filter" in queryplan:
            parsedPlan += " and bind them with the following condition(s) " + queryplan["Filter"].replace("::text", "")
        parsedPlan += "."
        return parsedPlan

    if queryplan["Strategy"] == "Hashed":
        text = Annotation.getConjuction()

        if len(queryplan["Group Key"]) == 1:
            text += "All rows hashed based on the keys "
            text += queryplan["Group Key"][0].replace("::text", "") + ", "
        else:
            text += "All rows hashed based on the keys "
            for i in queryplan["Group Key"]:
                text += i.replace("::text", "") + ", "
        text += " and after processing, the desired row is returned"

        parsedPlan = Annotation.parsePlan(queryplan["Plans"][0], isStart)
        parsedPlan += " " + text
        return parsedPlan

    if queryplan["Strategy"] == "Plain":
        parsedPlan = Annotation.parsePlan(queryplan["Plans"][0], isStart) + " "
        parsedPlan += Annotation.getConjuction()
        parsedPlan += "Aggregate the result."
        return parsedPlan

