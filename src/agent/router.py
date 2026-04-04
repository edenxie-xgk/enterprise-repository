from langgraph.constants import END

reasoning_route = [
    "rewrite_query",
    "expand_query",
    "decompose_query",
]

tool_route = [
    'rag',
]

route_list = reasoning_route + tool_route + ["abort","finish"]

route_map = {

}

for route in route_list:
    if route == 'finish':
        route_map[route] = END
    elif route == 'abort':
        route_map[route] = END
    else:
        route_map[route] = route