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
FCPW ray intersection: 33643us
SNCH-LBVH ray intersection: 581.438us
Case: example/2d/waker.obj
FCPW closest primitive: 43232.5us
SNCH-LBVH closest primitive: 1983.84us
FCPW closest silhouette: 55201.6us
SNCH-LBVH closest silhouette: 1579.19us
FCPW ray intersection: 37637.3us
SNCH-LBVH ray intersection: 1611.75us
Case: example/3d/suzanne.obj
FCPW closest primitive: 57444.8us
SNCH-LBVH closest primitive: 925.25us
FCPW closest silhouette: 122663us
SNCH-LBVH closest silhouette: 2691.28us
FCPW ray intersection: 40679.3us
SNCH-LBVH ray intersection: 972.75us
Case: example/3d/suzanne_subdiv.obj
FCPW closest primitive: 74262.2us
SNCH-LBVH closest primitive: 1772.72us
FCPW closest silhouette: 230439us
SNCH-LBVH closest silhouette: 5577.19us
FCPW ray intersection: 41190.4us
SNCH-LBVH ray intersection: 1087.69us
Case: example/3d/bunny.obj
FCPW closest primitive: 94780us
SNCH-LBVH closest primitive: 4390.81us
FCPW closest silhouette: 408931us
SNCH-LBVH closest silhouette: 16327.9us
FCPW ray intersection: 43646.7us
SNCH-LBVH ray intersection: 1168.62us
Case: example/3d/armadillo.obj
FCPW closest primitive: 102476us
SNCH-LBVH closest primitive: 5574.84us
FCPW closest silhouette: 299833us
SNCH-LBVH closest silhouette: 25080.9us
FCPW ray intersection: 44601.4us
SNCH-LBVH ray intersection: 1097.75us
Case: example/3d/kitten.obj
FCPW closest primitive: 200742us
SNCH-LBVH closest primitive: 12288.3us
FCPW closest silhouette: 502106us
SNCH-LBVH closest silhouette: 20612.9us
FCPW ray intersection: 43241us
SNCH-LBVH ray intersection: 1091.75us
"""

# Parsing the data
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

# Preparing data for plotting
filenames = [case['filename'] for case in cases]
metrics = ['closest_primitive', 'closest_silhouette', 'ray_intersection']
methods = ['FCPW', 'SNCH-LBVH']
combinations = [(method, metric) for metric in metrics for method in methods]
colors = {
    ('FCPW', 'closest_primitive'): '#1f77b4',
    ('SNCH-LBVH', 'closest_primitive'): '#003f5c',
    ('FCPW', 'closest_silhouette'): '#ff7f0e',
    ('SNCH-LBVH', 'closest_silhouette'): '#d62728',
    ('FCPW', 'ray_intersection'): '#2ca02c',
    ('SNCH-LBVH', 'ray_intersection'): '#006400'
}

# Plotting bar chart
x = np.arange(len(filenames))
total_bars = len(combinations)
bar_width = 0.13
offset = (total_bars / 2) * bar_width

fig, ax1 = plt.subplots(figsize=(10, 8))

# Adding background shading for alternating cases
for i in range(len(filenames)):
    if i % 2 == 0:
        ax1.axvspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.5)
    else:
        ax1.axvspan(i - 0.5, i + 0.5, facecolor='white', alpha=1.0)

# Adding bars
for i, (method, metric) in enumerate(combinations):
    values = [case[f'{method}_{metric}'] for case in cases]
    positions = x - offset + i * bar_width + bar_width / 2
    ax1.bar(positions, values, width=bar_width, label=f'{method} {metric.replace("_", " ").title()}', color=colors[(method, metric)])

ax1.set_xlabel('Case Name', fontsize=12)
ax1.set_ylabel('Time (us)', fontsize=12)
ax1.set_title('Benchmark Results Comparison with Speedup Ratios', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(filenames, rotation=45, ha='right')

# Creating second y-axis for speedup ratios
ax2 = ax1.twinx()
ax2.set_ylabel('Speedup Ratio', fontsize=12)

# Calculating and plotting speedup ratios
for i, metric in enumerate(metrics):
    speedup_ratios = [case[f'FCPW_{metric}'] / case[f'SNCH-LBVH_{metric}'] for case in cases]
    ax2.plot(x, speedup_ratios, marker='o', label=f'Speedup Ratio ({metric.replace("_", " ").title()})', linestyle='--')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()
fig.savefig('benchmark.png', dpi=300)
