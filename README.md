### Routing api

1. Giải routing với topo G và traffic matrix tm

    ```python
    from segment_routing import SegmentRoutingSolver

    solver = SegmentRoutingSolver(G)
    solver.solve(tm)
    ```

2. Tính utilization trên traffic matrix đã được giải

    ```python
    import util

    u = util.get_max_utilization(solver, tm)
    print(u)
    ```

3. Tính utilization trên traffic matrix mới, sử dụng lại routing đã được giải từ trước 

    ```python
    u = util.get_max_utilization_v2(solver, new_tm)
    print(u)
    ```

4. Liệt kê các đường đi từ đỉnh i đến j

    ```python
    paths = solver.get_paths(i, j)
    for k, path in paths:
        print(k, path)
    ```
    Lưu ý là hàm trả về list của các tuple gồm index của middle point và đường đi

5. Tính degree của mỗi node 
    
    ```python
    degree = util.get_degree(G, i)
    ```

6. Lấy danh sách các nodes sorted by degrees

    ```python
    nodes, degrees = util.get_nodes_sort_by_degree(G)
    for node, degree in zip(nodes, degrees):
        print(node, degree)
    ```

7. Lấy danh sách các flows đi qua từng node

    ```python
    from pprint import pprint

    node2flows = util.get_node2flows(solver)
    pprint(node2flows)

    for node in node2flows:
        print('node={} number of flow={}'.format(node, len(node2flows[node])))
    ```
