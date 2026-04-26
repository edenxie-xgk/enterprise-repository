from langgraph.constants import END

from src.agent.action_registry import REASONING_ACTION_NAMES, ROUTE_ACTION_NAMES, TOOL_ACTION_NAMES

reasoning_route = list(REASONING_ACTION_NAMES)
tool_route = list(TOOL_ACTION_NAMES)

route_list = list(ROUTE_ACTION_NAMES)
route_map = {}

for route in route_list:
    if route == 'finish':
        route_map[route] = END
    elif route == 'abort':
        route_map[route] = END
    else:
        route_map[route] = route
