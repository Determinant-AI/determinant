import yaml
from collections import deque
from python.async_app_fan import *

# Parse the YAML configuration


def parse_model_composition_conf(path: str):
    file = open(path, "r").read()
    config = yaml.safe_load(file)
    models_config = config["models"]

    # Create a dictionary to store instantiated objects
    model_objects = {}

    # Instantiate objects and construct the deployment graph
    for model_config in models_config:
        class_name = model_config["class_name"]
        # function_name = model_config["function"]
        dependencies = model_config.get("dependencies", [])
        inistialize_config = model_config.get("intialize_args", {})
        deploy_config = model_config.get("deploy_args", {})

        # Construct a dependency graph
        graph = {}
        indegree = {}
        for model_config in models_config:
            class_name = model_config["class_name"]
            dependencies = model_config.get("dependencies", [])
            graph[class_name] = dependencies
            indegree[class_name] = len(dependencies)

    # Topological sorting
    queue = deque()
    for node, degree in indegree.items():
        if degree == 0:
            queue.append(node)

    zero_degrees = [n for n in queue]

    sorted_order = []
    while queue:
        curr_node = queue.popleft()
        sorted_order.append(curr_node)
        for neighbor in graph:
            if curr_node in graph[neighbor]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

    # print(sorted_order)
    # Dynamically instantiate the class using the class name

    kwargs_to_bind = {}
    for n in zero_degrees:
        obj = globals()[n].bind()
        kwargs_to_bind[n] = obj

    for class_name in sorted_order:
        dep_kwargs = {dep_class: kwargs_to_bind[dep_class]
                      for dep_class in graph[class_name]}
        kwargs_to_bind[class_name] = globals()[class_name].bind(**dep_kwargs)

    return kwargs_to_bind


print(parse_model_composition_conf(
    "/Users/fanpan/workspace/ray/determinant/deployment/model-composition.yaml"))
