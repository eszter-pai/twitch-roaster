import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load HateBERT results
with open('test_classifiers/results/hatebert_results.json', 'r', encoding='utf-8') as f:
    hatebert_data = json.load(f)

# Create img directory if it doesn't exist
img_dir = Path('img')
img_dir.mkdir(exist_ok=True)

# Extract data
messages = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in hatebert_data['results']]
confidences = [r['confidence'] * 100 for r in hatebert_data['results']]
is_hateful = [r['is_hateful'] for r in hatebert_data['results']]

# Figure 1: Confidence scores bar chart
fig, ax = plt.subplots(figsize=(16, 10))
colors = ['#e74c3c' if h else '#2ecc71' for h in is_hateful]
bars = ax.barh(range(len(messages)), confidences, color=colors, alpha=0.7, edgecolor='black')

# Add threshold line
ax.axvline(x=50, color='black', linestyle='--', linewidth=2, alpha=0.5)

ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=9)
ax.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax.set_title('HateBERT Classification Results - All Messages Flagged as Hateful', 
             fontsize=14, fontweight='bold', pad=20)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label=f'Offensive ({hatebert_data["hateful_count"]})'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label=f'Not Offensive ({hatebert_data["not_hateful_count"]})'),
                   plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='50% Threshold')]
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, 100)

# Add value labels on bars
for i, (bar, conf) in enumerate(zip(bars, confidences)):
    ax.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(img_dir / 'hatebert_confidence_scores.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'hatebert_confidence_scores.png'}")
plt.close()

# Figure 2: Distribution of confidence scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(confidences, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax.axvline(x=np.mean(confidences), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(confidences):.1f}%')
ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
ax.set_title('HateBERT Confidence Score Distribution', fontsize=14, fontweight='bold', pad=15)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'hatebert_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'hatebert_distribution.png'}")
plt.close()

# Figure 3: Summary statistics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Pie chart - Classification results
labels = ['Hateful', 'Not Hateful']
sizes = [hatebert_data['hateful_count'], hatebert_data['not_hateful_count']]
colors_pie = ['#e74c3c', '#2ecc71']
explode = (0.1, 0)
ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Classification Results', fontsize=13, fontweight='bold', pad=15)

# Box plot - Confidence distribution
ax2.boxplot(confidences, vert=True, patch_artist=True,
            boxprops=dict(facecolor='#e74c3c', alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5))
ax2.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_ylabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticklabels(['All Messages'])
ax2.grid(axis='y', alpha=0.3)

# Statistics text
stats_text = f"""
Model: {hatebert_data['model']}

Total Messages: {hatebert_data['total_messages']}
Hateful: {hatebert_data['hateful_count']} (100%)
Not Hateful: {hatebert_data['not_hateful_count']} (0%)

Confidence Statistics:
  Mean: {np.mean(confidences):.2f}%
  Median: {np.median(confidences):.2f}%
  Min: {np.min(confidences):.2f}%
  Max: {np.max(confidences):.2f}%
  Std Dev: {np.std(confidences):.2f}%

⚠️ False Positive Rate: Very High
(All messages flagged including 
"hello hows your day" and "nice play!")
"""
ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5))
ax3.axis('off')
ax3.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=15)

# Confidence ranges
ranges = {
    '50-60%': sum(1 for c in confidences if 50 <= c < 60),
    '60-70%': sum(1 for c in confidences if 60 <= c < 70),
    '70-80%': sum(1 for c in confidences if 70 <= c < 80),
    '80-90%': sum(1 for c in confidences if 80 <= c < 90),
    '90-100%': sum(1 for c in confidences if 90 <= c <= 100)
}
ax4.bar(ranges.keys(), ranges.values(), color='#e74c3c', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Confidence Range', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
ax4.set_title('Messages by Confidence Range', fontsize=13, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)
for i, (k, v) in enumerate(ranges.items()):
    ax4.text(i, v + 0.3, str(v), ha='center', fontweight='bold')

plt.suptitle('HateBERT Classifier Analysis - High False Positive Rate', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(img_dir / 'hatebert_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'hatebert_summary.png'}")
plt.close()

# Figure 4: Top 10 highest and lowest confidence
sorted_indices = np.argsort(confidences)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Lowest 10
low_indices = sorted_indices[:10]
low_messages = [messages[i] for i in low_indices]
low_confs = [confidences[i] for i in low_indices]
ax1.barh(range(len(low_messages)), low_confs, color='#f39c12', alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(low_messages)))
ax1.set_yticklabels(low_messages, fontsize=9)
ax1.set_xlabel('Confidence Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('10 Lowest Confidence Scores\n(Still flagged as hateful)', 
              fontsize=12, fontweight='bold', pad=15)
ax1.axvline(x=50, color='black', linestyle='--', linewidth=2, alpha=0.5)
for i, conf in enumerate(low_confs):
    ax1.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=8)

# Highest 10
high_indices = sorted_indices[-10:]
high_messages = [messages[i] for i in high_indices]
high_confs = [confidences[i] for i in high_indices]
ax2.barh(range(len(high_messages)), high_confs, color='#c0392b', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(high_messages)))
ax2.set_yticklabels(high_messages, fontsize=9)
ax2.set_xlabel('Confidence Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('10 Highest Confidence Scores', fontsize=12, fontweight='bold', pad=15)
for i, conf in enumerate(high_confs):
    ax2.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(img_dir / 'hatebert_top_bottom.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'hatebert_top_bottom.png'}")
plt.close()

print(f"\n🎨 All visualizations saved to {img_dir}/")
