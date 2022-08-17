# Copyright 2022 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Iterable

import graphviz


class UndirectedGraph:
    """An undirected graph datastructure."""

    def __init__(self, nodes: Iterable, edges: Iterable):
        """Create the graph by specifying nodes and edges explicitly.
        Args:
            nodes (iterable): The nodes.
            edges (iterable): The edges as a list of pairs of nodes.
        """
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.check_nodes_edges()

    def check_nodes_edges(self):
        for e1, e2 in self.edges:
            edge = (e1, e2)
            if e1 not in self.nodes:
                raise ValueError("{} from {} not in nodes".format(e1, edge))
            if e2 not in self.nodes:
                raise ValueError("{} from {} not in nodes".format(e2, edge))
            if (e2, e1) in self.edges:
                raise ValueError("Duplicate edge {}".format(edge))

    def visualize(self):
        """Visualize the graph with Graphviz.
        Returns:
            Graphviz object: The graph as a Graphviz object.
        """
        g = graphviz.Graph()
        for e1, e2 in self.edges:
            g.edge(str(e1), str(e2))
        for node in self.nodes:
            g.node(str(node))
        return g

    def from_edge_list(edges):
        """Create an undirected graph from a list of edges.
        Args:
            edges (list): List of tuples forming the edges.
        Returns:
            UndirectedGraph: The resulting UndirectedGraph.
        """
        nodes = set()
        for e1, e2 in edges:
            nodes.add(e1)
            nodes.add(e2)
        return UndirectedGraph(nodes, edges)

    def from_clique_list(cliques):
        """Create an undirected graph from a list of cliques.
        Args:
            cliques (list): List of cliques as a list of tuples of nodes.
        Returns:
            UndirectedGraph: The resulting UndirectedGraph.
        """
        nodes = set()
        edges = set()
        for clique in cliques:
            for i, node in enumerate(clique):
                nodes.add(node)
                for j in range(i + 1, len(clique)):
                    edges.add((node, clique[j]))

        return UndirectedGraph(nodes, edges)

    def copy(self):
        nodes = self.nodes.copy()
        edges = self.edges.copy()
        return UndirectedGraph(nodes, edges)

    def add_edge(self, edge):
        e1, e2 = edge
        if (e2, e1) not in self.edges:
            self.edges.add(edge)
            self.add_node(e1)
            self.add_node(e2)

    def add_node(self, node):
        self.nodes.add(node)

    def get_neighbours(self, node):
        return [
            other
            for other in self.nodes
            if (node, other) in self.edges or (other, node) in self.edges
        ]


def node_ordering_cost(graph: UndirectedGraph, node: Iterable):
    return len(graph.get_neighbours(node))


def greedy_ordering(graph: UndirectedGraph) -> List:
    """Find a greedy ordering for variable elimination.
    Args:
        graph (UndirectedGraph): The graph to find the ordering on.
    Returns:
        list: The elimination order as a list of nodes.
    """
    graph = graph.copy()
    unmarked_nodes = set(graph.nodes)
    ordering = []
    for i in range(len(unmarked_nodes)):
        min_cost = 2**32
        min_cost_node = None
        for node in unmarked_nodes:
            cost = node_ordering_cost(graph, node)
            if cost < min_cost:
                min_cost = cost
                min_cost_node = node

        ordering.append(min_cost_node)
        unmarked_nodes.remove(min_cost_node)

        neighbours = graph.get_neighbours(min_cost_node)
        for i, neighbour in enumerate(neighbours):
            for other_neighbour in neighbours[i + 1 :]:
                graph.add_edge((neighbour, other_neighbour))

    return ordering
