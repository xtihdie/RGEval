from pyvis.network import Network

data = {'1.1': [1], '1.2': [2], '1.3': [1], '1.4': [3], '1.5': [4, 5], '1.6': [5, 6], '1.7': [6],
        '1.8': [7], '1.9': [7], '1.10': [8, 15], '1.11': [16, 9], '1.12': [17, 10], '1.13': [18, 11],
        '1.14': [19, 12], '1.15': [19, 12, 13], '1.16': [20, 13], '1.17': [14], '1.18': [1, 20, 21]}


# 创建网络
net = Network(height="900px", width="100%", directed=True)

# 提取所有唯一值节点，并按1,2,3,4...的顺序排列
value_set = set()
for sublist in data.values():
    value_set.update(sublist)
value_nodes = [str(i) for i in range(1, 18) if i in value_set]  # 按顺序排列1到17的值

# 对键节点排序
key_nodes = sorted(data.keys(), key=lambda x: [int(n) for n in x.split('.')])

# 创建一级节点并添加
# 提取一级节点标识 (如1, 2等，实际本例只有'1')
level1_ids = sorted(set(key.split('.')[0] for key in data.keys()))

# 存储一级节点的x坐标
level1_node_x = {}
for lvl1 in level1_ids:
    # 修改：一级节点使用特殊前缀"L1_"避免ID冲突
    net.add_node(f"L1_{lvl1}",
                 x=0, y=-450,  # 临时位置，后面会调整
                 color='#BB1111',
                 size=40,
                 title=f"<div style='margin-top:15px;'>Level 1: {lvl1}</div>",
                 level=0)
    level1_node_x[lvl1] = None  # 初始化

# 创建二级节点并添加
# 提取二级节点标识 (如1.1, 1.2等)
group_ids = sorted(set('.'.join(key.split('.')[:2]) for key in data.keys()))
group_node_x = {}

# 存储每个一级节点包含的二级节点
lvl1_subgroups = {lvl1: [] for lvl1 in level1_ids}

# 添加二级节点（水平排列在上方）
for i, group in enumerate(group_ids):
    x_pos = i * 600 + 1000
    # 修改：二级节点使用特殊前缀"L2_"避免ID冲突
    net.add_node(f"L2_{group}", x=x_pos, y=-150,
                 color='#00AAAA',
                 size=30,
                 title=f"<div style='margin-top:15px;'>Level 2: {group}</div>",
                 level=1)
    group_node_x[group] = x_pos
    # 记录属于哪个一级节点
    lvl1_id = group.split('.')[0]
    lvl1_subgroups[lvl1_id].append(group)

# 更新一级节点位置（放置在所有二级节点的中心正上方）
for lvl1_id, subgroups in lvl1_subgroups.items():
    if subgroups:
        x_coords = [group_node_x[g] for g in subgroups]
        center_x = (min(x_coords) + max(x_coords)) / 2
        # 更新一级节点位置
        level1_node_x[lvl1_id] = center_x
        # 修改：使用新的ID前缀
        net.get_node(f"L1_{lvl1_id}")['x'] = center_x

# 添加一级节点到二级节点的边
for lvl1_id in level1_ids:
    for subgroup in lvl1_subgroups[lvl1_id]:
        # 修改：使用新的ID前缀
        net.add_edge(f"L1_{lvl1_id}", f"L2_{subgroup}",
                     width=4, color='#FF5555', arrows='to')

# 添加键节点（水平排列，根据所属的二级分组）
key_node_x = {}
for i, key in enumerate(key_nodes):
    group = '.'.join(key.split('.')[:2])
    # 计算同组键节点中的位置
    group_keys = [k for k in key_nodes if k.startswith(group+'.')]
    idx = group_keys.index(key)
    total = len(group_keys)
    spacing = 150
    x = spacing * i
    # 将标题放在节点下方
    title_html = f"<div style='margin-top:15px;'>{key}</div>"
    net.add_node(key, x=x, y=0, color='#FFFF00', size=25,
                title=title_html, level=2)
    key_node_x[key] = x
    # 添加二级节点到键节点的边 - 修改：使用新的ID前缀
    net.add_edge(f"L2_{group}", key, width=3, color='#00CCAA', arrows='to')

# 添加值节点（水平排列，按1,2,3,4顺序）
for i, value in enumerate(value_nodes):
    # 将标题放在节点下方
    title_html = f"<div style='margin-top:15px;'>{value}</div>"
    net.add_node(value, x=i * 350 + 550, y=200, color='#5555FF', size=20,
                title=title_html, level=3)

# 添加数据边（键节点到值节点）
for key, values in data.items():
    for value in values:
        if value:  # 确保值存在
            net.add_edge(key, str(value), width=2, color='#AAAA00')

# 禁用物理引擎保持布局
net.set_options("""
{
  "physics": {
    "enabled": false
  },
  "edges": {
    "color": "#FFA500",
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 0.5
      }
    },
    "smooth": false
  },
  "nodes": {
    "font": {
      "size": 14,
      "align": "center"
    },
    "labelHighlightBold": false,
    "scaling": {
      "min": 10,
      "max": 30
    }
  },
  "interaction": {
    "tooltipDelay": 0,
    "hover": true
  }
}
""")

# 保存为HTML文件
net.show("graph.html", notebook=False)