# DSU class
class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n)]  # Each node is its own parent initially

    # Find with path compression
    def find_parent(self, u):
        if u == self.parent[u]:
            return u
        self.parent[u] = self.find_parent(self.parent[u])
        return self.parent[u]

    # Union the sets of u and v
    def unite(self, u, v):
        pu = self.find_parent(u)
        pv = self.find_parent(v)
        if pu == pv:
            return
        self.parent[pu] = pv


# Function to find all connected components using DSU
def getComponents(adj):
    V = len(adj)
    dsu = DSU(V)

    # unite components on the basis of edges
    for i in range(V):
        for nxt in adj[i]:
            dsu.unite(i, nxt)

    res_map = {}
    for i in range(V):
        par = dsu.find_parent(i)
        res_map.setdefault(par, []).append(i)

    res = list(res_map.values())
    return res
