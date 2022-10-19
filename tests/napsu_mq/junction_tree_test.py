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


import unittest
from twinify.napsu_mq.junction_tree import *
from twinify.napsu_mq.undirected_graph import greedy_ordering, UndirectedGraph


class JunctionTreeTest(unittest.TestCase):
    def setUp(self):
        self.query_sets1 = [
            ("C", "D"), ("D", "G"), ("I", "G"), ("I", "S"), ("S", "J"), ("G", "L"), ("G", "H"), ("L", "J")
        ]
        self.graph1 = UndirectedGraph.from_edge_list(self.query_sets1)
        self.order1 = greedy_ordering(self.graph1)
        self.jt1 = JunctionTree.from_variable_elimination(self.query_sets1, self.order1, enable_checks=True)
        self.jt1.remove_redundant_nodes()

        self.query_sets2 = [
            ("edu", "income"), ("hours", "income"), ("loss", "income"), ("gain", "income"), ("race", "income"),
            ("gender", "income"), ("race", "gender"), ("age", "income"), ("age", "workclass"), ("age", "mamrital")
        ]
        self.graph2 = UndirectedGraph.from_edge_list(self.query_sets2)
        self.order2 = greedy_ordering(self.graph2)
        self.jt2 = JunctionTree.from_variable_elimination(self.query_sets2, self.order2, enable_checks=True)
        self.jt2.remove_redundant_nodes()

        self.query_sets3 = [
            ("C", "D"), ("D", "G"), ("I", "G"), ("I", "S", "G"), ("S", "J"), ("G", "L", "H", "J"), ("L", "J")
        ]
        self.graph3 = UndirectedGraph.from_clique_list(self.query_sets3)
        self.order3 = greedy_ordering(self.graph3)
        self.jt3 = JunctionTree.from_variable_elimination(self.query_sets3, self.order3, enable_checks=True)
        self.jt3.remove_redundant_nodes()

        self.query_sets4 = [
            ("A", "B"), ("C", "D"), ("D", "G"), ("I", "G"), ("I", "S", "G"), ("S", "J"), ("G", "L", "H", "J"),
            ("L", "J")
        ]
        self.graph4 = UndirectedGraph.from_clique_list(self.query_sets4)
        self.order4 = greedy_ordering(self.graph4)
        self.jt4 = JunctionTree.from_variable_elimination(self.query_sets4, self.order4, enable_checks=True)
        self.jt4.remove_redundant_nodes()

        self.query_sets5 = [(0, 3), (1, 3), (2, 3), (0, 1)]
        self.graph5 = UndirectedGraph.from_clique_list(self.query_sets5)
        self.order5 = greedy_ordering(self.graph5)
        self.jt5 = JunctionTree.from_variable_elimination(self.query_sets5, self.order5, enable_checks=True)
        self.jt5.remove_redundant_nodes()

    def test_nodes_edges(self):
        self.jt1.check_nodes_edges()
        self.jt2.check_nodes_edges()
        self.jt3.check_nodes_edges()
        self.jt4.check_nodes_edges()
        self.jt5.check_nodes_edges()

    def test_acyclicity(self):
        self.jt1.check_acyclicity()
        self.jt2.check_acyclicity()
        self.jt3.check_acyclicity()
        self.jt4.check_acyclicity()
        self.jt5.check_acyclicity()

    def check_redundant_nodes(self, jt):
        nodes = list(jt.nodes)
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                self.assertFalse(set(node1).issubset(node2))
                self.assertFalse(set(node2).issubset(node1))

    def test_redundant_nodes(self):
        self.check_redundant_nodes(self.jt1)
        self.check_redundant_nodes(self.jt2)
        self.check_redundant_nodes(self.jt3)
        self.check_redundant_nodes(self.jt4)

    def check_factors(self, jt, scopes):
        for scope in scopes:
            is_subset = any([set(scope).issubset(node) for node in jt.nodes])
            self.assertTrue(is_subset)

    def test_factors(self):
        self.check_factors(self.jt1, self.query_sets1)
        self.check_factors(self.jt2, self.query_sets2)
        self.check_factors(self.jt3, self.query_sets3)
        # TODO: fix juction tree reduction for disconnected trees
        # self.check_factors(self.jt3, self.query_sets4)

    def check_running_intersection(self, jt):
        def check_subtree(node, parent, variable, can_see):
            if not can_see:
                self.assertFalse(variable in node)
                new_can_see = False
            else:
                new_can_see = variable in node
            neighbours = jt.get_neighbours(node)
            for neighbour in neighbours:
                if neighbour != node and neighbour != parent:
                    check_subtree(neighbour, node, variable, new_can_see)

        tested_variables = set()
        for node in jt.nodes:
            for variable in node:
                if variable not in tested_variables:
                    check_subtree(node, None, variable, True)
                    tested_variables.add(variable)

    def test_running_intersection(self):
        self.check_running_intersection(self.jt1)
        self.check_running_intersection(self.jt2)
        self.check_running_intersection(self.jt3)
        self.check_running_intersection(self.jt4)

    def check_node_for_factor(self, jt, query_sets):
        for scope in query_sets:
            self.assertTrue(set(scope).issubset(jt.node_for_factor[scope]))

    def test_node_for_factor(self):
        self.check_node_for_factor(self.jt1, self.query_sets1)
        self.check_node_for_factor(self.jt2, self.query_sets2)
        self.check_node_for_factor(self.jt3, self.query_sets3)
        self.check_node_for_factor(self.jt4, self.query_sets4)

    def check_factors_in_node(self, jt):
        for node, factors in jt.factors_in_node.items():
            for factor in factors:
                self.assertEqual(jt.node_for_factor[factor], node)

    def test_factor_in_node(self):
        self.check_factors_in_node(self.jt1)
        self.check_factors_in_node(self.jt2)
        self.check_factors_in_node(self.jt3)
        self.check_factors_in_node(self.jt4)

    def check_downward_order(self, jt):
        self.assertSetEqual(set(node.variables for node in jt.downward_order), jt.nodes)
        marked_nodes = set()
        for node in jt.downward_order:
            if node.parent is not None:
                self.assertTrue(node.parent in marked_nodes)
            marked_nodes.add(node)

    def check_upward_order(self, jt):
        marked_nodes = set()
        for node in jt.upward_order:
            for child in node.children:
                self.assertTrue(child in marked_nodes)
            marked_nodes.add(node)

    def check_children(self, jt):
        marked_nodes = set()
        node_stack = [jt.root_node]
        while len(node_stack) > 0:
            node = node_stack.pop()
            marked_nodes.add(node.variables)
            for child in node.children:
                self.assertTrue(child.variables not in marked_nodes)
                node_stack.append(child)
        self.assertSetEqual(marked_nodes, set(jt.nodes))

    def test_downward_order(self):
        self.check_downward_order(self.jt1)
        self.check_downward_order(self.jt2)
        self.check_downward_order(self.jt3)
        # self.check_downward_order(self.jt4)

    def test_upward_order(self):
        self.check_upward_order(self.jt1)
        self.check_upward_order(self.jt2)
        self.check_upward_order(self.jt3)
        # self.check_upward_order(self.jt4)

    def test_children(self):
        self.check_children(self.jt1)
        self.check_children(self.jt2)
        self.check_children(self.jt3)
        # self.check_children(self.jt4)

    def test_root_node(self):
        self.assertTupleEqual(self.jt1.root_node.variables, self.jt1.root_variables)
        self.assertTupleEqual(self.jt2.root_node.variables, self.jt2.root_variables)
        self.assertTupleEqual(self.jt3.root_node.variables, self.jt3.root_variables)
        self.assertTupleEqual(self.jt4.root_node.variables, self.jt4.root_variables)


if __name__ == "__main__":
    unittest.main()
