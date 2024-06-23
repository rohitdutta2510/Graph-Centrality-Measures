import numpy as np
from collections import deque

def get_citation_graph(file_path):
    adjacency_list = {}
    node_list = set()

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into source and target vertices
            target, source = map(int, line.strip().split())
            # Add source and target nodes to the list containing all the graph nodes
            node_list.add(source)
            node_list.add(target)

            # Add the edge to the adjacency list
            if source in adjacency_list:
                adjacency_list[source].append(target)
            else:
                adjacency_list[source] = [target]

    return adjacency_list, node_list

def add_sink_nodes(graph, node_list):
    graph_nodes = list(graph.keys())
    sink_nodes = []
    # Identify and extract the sink nodes from the node list
    for node in node_list:
        if node not in graph_nodes:
            sink_nodes.append(node)

    # Sink nodes modification 
    for node in sink_nodes:
        add_nodes = []
        for key, values in graph.items():
            if node in values:
                add_nodes.append(key)
        graph[node] = add_nodes

    return graph

def find_shortest_paths(start, end, graph, dist):
    # Base condition
    if start == end:
        return [[start]]
    
    paths = [] # list to store all the shortest path from start to end
    for neighbor in graph[start]:
        if dist[start][end] == dist[neighbor][end] + 1:
            for path in find_shortest_paths(neighbor, end, graph, dist):
                paths.append([start] + path)
    
    return paths

def get_shortest_path_distances(graph, start):
    # Perform BFS
    # Initialize distances dictionary to store the shortest distances
    distances = {node: float('inf') for node in graph}
    
    # Set the distance to the start node as 0
    distances[start] = 0
    queue = deque([start])

    while queue:
        current_node = queue.popleft()

        for adj_node in graph[current_node]:
            if distances[adj_node] == float('inf'):
                distances[adj_node] = distances[current_node] + 1
                queue.append(adj_node)

    return distances

def all_shortest_paths(graph):
    INF = float('inf')
    dist = {}
    nodeList = list(graph.keys())
    
    # Iterate through all nodes and find shortest paths from each node
    print('>> Retrieving shortest path distances between every node pair (i, j)\n')
    for node in graph:
        dist[node] = get_shortest_path_distances(graph, node)

    # Extract all shortest paths between every pair of nodes
    print('>> Extracting all the shortest paths between every node pair (i, j) \n')

    # Initialize a dictionary of dictionaaries to store all shortest path between i and j
    paths = {}
    for node in nodeList:
        temp = {}
        for adj in nodeList:
            temp[adj] = []
        paths[node] = temp

    for i in nodeList:
        for j in nodeList:
            if dist[i][j] != INF:
                paths[i][j] = find_shortest_paths(i, j, graph, dist)
    
    return paths, dist

def get_closeness_centrality(distance_dict):
    n = len(distance_dict) # get number of nodes in the graph
    centrality_scores = {} # dictionary to store the closeness centrality of the nodes

    for node, distances in distance_dict.items(): # node -> node id, distanes -> dictionary (node id, distance dictionary)
        for key, value in distances.items(): # key -> node id, value -> distance from node to key
            # If the distance between a pair of nodes is 'infinity' (unreachable) mark them as '0' to avoid getting 'infinity' in the sum of distances
            if value == float('inf'):
                distance_dict[node][key] = 0
                
        total_distance = sum(list(distance_dict[node].values()))
        closeness_centrality = (n - 1) / total_distance if total_distance != 0 else 0
        centrality_scores[node] = closeness_centrality 

    # Sort the closeness centrality scores in descending order
    centrality_scores = {k: v for k, v in sorted(centrality_scores.items(), key=lambda item: item[1], reverse = True)}

    return centrality_scores

def get_betweenness_centrality(shortest_path_list):
    nodeList = list(shortest_path_list.keys()) # get all the nodes in graph
    n = len(nodeList) # get number of nodes in the graph
    bet_centrality = {}

    for node in nodeList:
        centrality = 0
        for i, pathList in shortest_path_list.items(): # list of all shortest paths for each node i to all node j
            for j, shortest_paths in pathList.items(): # list of all shortest paths from node i to node j
                tot_path_i_j = len(shortest_paths)
                i_node_j = 0 
                for path in shortest_paths: # one of the shortest path from i to j
                    # check if node is in the path i -> j and it itself is not the node i or j
                    if node in path and node != i and node != j:
                        i_node_j += 1

                centrality += i_node_j / tot_path_i_j if tot_path_i_j != 0 else 0

        bet_centrality[node] = centrality / ((n-1) * (n-2))

    # Sort the betwenness centrality scores in descending order
    bet_centrality = {k: v for k, v in sorted(bet_centrality.items(), key=lambda item: item[1], reverse = True)}
    return bet_centrality

def get_pagerank(graph, damping_factor=0.8, epsilon=1e-8, max_iterations=100):
    nodeList = list(graph.keys()) # get the nodes of the graph
    num_nodes = len(nodeList) # get number of nodes in the graph
    node_dir = {}

    # Assign an index value to the nodes and store them for future reference
    i = 0
    for node in nodeList:
        node_dir[node] = i
        i += 1

    undamped_transition_mat = np.zeros((num_nodes, num_nodes))

    for node, adj in graph.items():
        if adj:
            num_adj = len(adj)
            for neighbor in adj:
                x = node_dir[neighbor] # get index value for neighbor
                y = node_dir[node] # get index value for node
                undamped_transition_mat[x, y] = 1 / num_adj

    # Damping factor
    damping_matrix = (1 - damping_factor) / num_nodes * np.ones((num_nodes, num_nodes))
    transition_matrix = damping_factor * undamped_transition_mat + damping_matrix

    # Initial PageRank values
    page_rank_vector = np.ones(num_nodes) / num_nodes

    # Power iteration method to calculate PageRank
    for i in range(max_iterations):
        new_page_rank = np.dot(transition_matrix, page_rank_vector)
        if np.linalg.norm(new_page_rank - page_rank_vector, ord=1) < epsilon:
            break
        page_rank_vector = new_page_rank

    page_rank_final = {} # dictionary to store node id and corresponding page rank
    for node in nodeList:
        page_rank_final[node] = page_rank_vector[node_dir[node]]

    # Sort the pagerank scores in descending order
    page_rank_final = {k: v for k, v in sorted(page_rank_final.items(), key=lambda item: item[1], reverse = True)}
    return page_rank_final

def write_file(file_name, data_dict):
    file_path = './centralities/' + file_name
    with open(file_path, 'w') as file:
        for node, centrality in data_dict.items():
            file.write('{} {:.6f}\n'.format(node, centrality))


def main():
    # file_path = './demo.txt'
    file_path = './cora/cora.cites'
    print('>> Fetching the graph\n')
    citation_network, node_list = get_citation_graph(file_path)
    print('>> Adding the sink nodes\n')
    citation_network = add_sink_nodes(citation_network, node_list)
    shortest_paths, dist = all_shortest_paths(citation_network)

    print('>> Computing closeness centrality\n')
    closeness_centrality = get_closeness_centrality(dist)
    write_file('closeness.txt', closeness_centrality)

    print('>> Computing betweenness centrality\n')
    betweenness_centrality = get_betweenness_centrality(shortest_paths)
    write_file('beetweenness.txt', betweenness_centrality)

    print('>> Computing pagerank centrality\n')
    pagerank = get_pagerank(citation_network)
    write_file('pagerank.txt', pagerank)

    print('>> Closing gen_centrality.py\n')


if __name__ == '__main__':
    print('\n>> Running gen_centrality.py\n')
    main()