# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

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
from twinify.napsu_mq.undirected_graph import *


class UndirectedGraphTest(unittest.TestCase):
    def setUp(self):
        self.query_sets = [
            ("C", "D"), ("D", "G"), ("I", "G"), ("I", "S"), ("S", "J"), ("G", "L"), ("G", "H"), ("L", "J")
        ]
        self.g = UndirectedGraph.from_clique_list(self.query_sets)

    def test_from_cliques_nodes(self):
        self.assertSetEqual({"C", "D", "G", "I", "G", "S", "J", "L", "H"}, self.g.nodes)

    def test_from_cliques_edges(self):
        for query_set in self.query_sets:
            self.assertIn(query_set, self.g.edges)

    def test_from_cliques_edges_3way(self):
        g = UndirectedGraph.from_clique_list(self.query_sets + [("I", "S", "L")])
        for query_set in self.query_sets + [("I", "L"), ("S", "L")]:
            self.assertIn(query_set, g.edges)

    def test_from_edge_list_edges(self):
        g = UndirectedGraph.from_edge_list(self.query_sets)
        for query_set in self.query_sets:
            self.assertIn(query_set, g.edges)

    def test_copy(self):
        g1 = self.g.copy()
        self.assertSetEqual(self.g.nodes, g1.nodes)
        self.assertSetEqual(self.g.edges, g1.edges)

    def test_get_neighbours(self):
        neighbours = self.g.get_neighbours("G")
        self.assertSetEqual(set(neighbours), {"D", "I", "L", "H"})

    def test_add_node(self):
        g1 = self.g.copy()
        g1.add_node("P")
        self.assertIn("P", g1.nodes)

    def test_add_edge(self):
        g1 = self.g.copy()
        g1.add_edge(("P", "G"))
        self.assertIn(("P", "G"), g1.edges)

    def test_node_ordering(self):
        ordering = greedy_ordering(self.g)
        self.assertSetEqual(set(ordering), self.g.nodes)

    def test_naive_bayes_ordering(self):
        g = UndirectedGraph.from_edge_list([(0, i) for i in range(1, 5)])
        ordering = greedy_ordering(g)
        self.assertEqual(ordering[-1], 0)


if __name__ == "__main__":
    unittest.main()
