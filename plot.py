import matplotlib.pyplot as plt
import re
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans']

data = """
Case: example/2d/circle_in_square.obj
FCPW closest primitive: 37625.3us
SNCH-LBVH closest primitive: 273.656us
FCPW closest silhouette: 45499.1us
SNCH-LBVH closest silhouette: 302.406us
FCPW ray intersection: 37644us
SNCH-LBVH ray intersection: 539.031us
Case: example/2d/waker.obj
FCPW closest primitive: 43232.5us
SNCH-LBVH closest primitive: 1983.84us
FCPW closest silhouette: 55201.6us
SNCH-LBVH closest silhouette: 1579.19us
FCPW ray intersection: 38564us
SNCH-LBVH ray intersection: 2353.84us
Case: example/3d/suzanne.obj
FCPW closest primitive: 57444.8us
SNCH-LBVH closest primitive: 925.25us
FCPW closest silhouette: 122663us
SNCH-LBVH closest silhouette: 2691.28us
FCPW ray intersection: 41547us
SNCH-LBVH ray intersection: 2467.28us
Case: example/3d/suzanne_subdiv.obj
FCPW closest primitive: 74262.2us
SNCH-LBVH closest primitive: 1772.72us
FCPW closest silhouette: 230439us
SNCH-LBVH closest silhouette: 5577.19us
FCPW ray intersection: 41751.2us
SNCH-LBVH ray intersection: 7232.19us
Case: example/3d/bunny.obj
FCPW closest primitive: 94780us
SNCH-LBVH closest primitive: 4390.81us
FCPW closest silhouette: 408931us
SNCH-LBVH closest silhouette: 16327.9us
FCPW ray intersection: 40203us
SNCH-LBVH ray intersection: 47789.2us
Case: example/3d/armadillo.obj
FCPW closest primitive: 102476us
SNCH-LBVH closest primitive: 5574.84us
FCPW closest silhouette: 299833us
SNCH-LBVH closest silhouette: 25080.9us
FCPW ray intersection: 42694.2us
SNCH-LBVH ray intersection: 83677.4us
Case: example/3d/kitten.obj
FCPW closest primitive: 200742us
SNCH-LBVH closest primitive: 12288.3us
FCPW closest silhouette: 502106us
SNCH-LBVH closest silhouette: 20612.9us
FCPW ray intersection: 41165.5us
SNCH-LBVH ray intersection: 176338us
"""

cases = []
current_case = {}
for line in data.strip().split('\n'):
    line = line.strip()
    if not line:
        continue
    case_match = re.match(r"Case:\s+(.+)", line)
    if case_match:
        if current_case:
            cases.append(current_case)
            current_case = {}
        filepath = case_match.group(1)
        filename = filepath.split('/')[-1].replace('_', ' ').replace('.obj', '')
        current_case['filename'] = filename
    else:
        match = re.match(r"(FCPW|SNCH-LBVH) (closest primitive|closest silhouette|ray intersection): ([\d\.]+)us", line)
        if match:
            method = match.group(1)
            metric = match.group(2).replace(' ', '_')
            value = float(match.group(3))
            current_case[f"{method}_{metric}"] = value
if current_case:
    cases.append(current_case)

filenames = [case['filename'] for case in cases]
metrics = ['closest_primitive', 'closest_silhouette', 'ray_intersection']
methods = ['FCPW', 'SNCH-LBVH']
combinations = [(method, metric) for metric in metrics for method in methods]
colors = {
    ('FCPW', 'closest_primitive'): '#1f77b4',       # Deep Blue
    ('SNCH-LBVH', 'closest_primitive'): '#003f5c', # Darker Blue
    ('FCPW', 'closest_silhouette'): '#ff7f0e',      # Deep Orange
    ('SNCH-LBVH', 'closest_silhouette'): '#d62728',# Darker Red
    ('FCPW', 'ray_intersection'): '#2ca02c',        # Deep Green
    ('SNCH-LBVH', 'ray_intersection'): '#006400'   # Darker Green
}

x = np.arange(len(filenames))
total_bars = len(combinations)
bar_width = 0.13
offset = (total_bars / 2) * bar_width

plt.figure(figsize=(10, 8))

ax = plt.gca()
for i in range(len(filenames)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.5)
    else:
        ax.axvspan(i - 0.5, i + 0.5, facecolor='white', alpha=1.0)

for i, (method, metric) in enumerate(combinations):
    values = [case[f'{method}_{metric}'] for case in cases]
    positions = x - offset + i * bar_width + bar_width / 2
    plt.bar(positions, values, width=bar_width, label=f'{method} {metric.replace("_", " ").title()}', color=colors[(method, metric)])
    # for idx, value in enumerate(values):
    #     plt.text(positions[idx], value + max(values)*0.005, f'{value:.1f}', ha='center', va='bottom', fontsize=7)

plt.xlabel('Case Name', fontsize=12)
plt.ylabel('Time (us)', fontsize=12)
plt.title('Benchmark Results Comparison', fontsize=16)
plt.xticks(x, filenames, rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig('benchmark.png', dpi=300)
