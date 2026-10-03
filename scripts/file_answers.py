ID_ACTIONS = ("LocateFile", "LocateCounterpart", "LocateSymbol", "LocateCodeSnippet", "SemanticSearch", "ShowCode", "LocateRelatedArtifacts")

def validate(answer, batch, stage):
    rows = answer.get("action_answers" if stage == 0 else "rankings")
    n = len(batch["queries"])
    if not isinstance(rows, list) or len(rows) != n:
        raise ValueError("Incomplete query answers")
    if sorted(x.get("query_id", -1) for x in rows) != list(range(n)):
        raise ValueError("Invalid query IDs")
    required = batch["calls"][0]["payload"].get("action_questions", {})
    for row in rows:
        limit = len(batch["queries"][row["query_id"]]["candidate_paths"])
        keys = [k for k in ID_ACTIONS if k in required] if stage == 0 else ["ranked_candidate_ids"]
        for key in keys:
            ids = row.get(key)
            if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < limit for i in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("Invalid candidate IDs")
            if stage == 1 and len(ids) < min(5, limit):
                raise ValueError("Incomplete final top-5 ranking")
