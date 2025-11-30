import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load LogReg results
with open('test_classifiers/results/logreg_results.json', 'r', encoding='utf-8') as f:
    logreg_data = json.load(f)

# Create img directory if it doesn't exist
img_dir = Path('img')
img_dir.mkdir(exist_ok=True)

# Extract data
messages = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in logreg_data['results']]
confidences = [r['confidence'] * 100 for r in logreg_data['results']]
is_offensive = [r['is_offensive'] for r in logreg_data['results']]
prob_offensive = [r['prob_offensive'] * 100 for r in logreg_data['results']]

# Figure 1: Confidence scores bar chart
fig, ax = plt.subplots(figsize=(16, 10))
colors = ['#e74c3c' if o else '#2ecc71' for o in is_offensive]
bars = ax.barh(range(len(messages)), confidences, color=colors, alpha=0.7, edgecolor='black')

# Add threshold line
ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold', alpha=0.5)

ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=9)
ax.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Logistic Regression Classifier Results - Conservative Approach', 
             fontsize=14, fontweight='bold', pad=20)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label=f'Offensive ({logreg_data["offensive_count"]}'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label=f'Not Offensive ({logreg_data["not_offensive_count"]}'),
                   plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='50% Threshold')]
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, 100)

# Add value labels on bars
for i, (bar, conf) in enumerate(zip(bars, confidences)):
    ax.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(img_dir / 'logreg_confidence_scores.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'logreg_confidence_scores.png'}")
plt.close()

# Figure 2: Distribution of confidence scores by classification
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

offensive_confs = [c for c, o in zip(confidences, is_offensive) if o]
not_offensive_confs = [c for c, o in zip(confidences, is_offensive) if not o]

ax1.hist(not_offensive_confs, bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
ax1.axvline(x=np.mean(not_offensive_confs), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(not_offensive_confs):.1f}%')
ax1.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax1.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
ax1.set_title(f'Not Offensive Distribution (n={len(not_offensive_confs)})', fontsize=13, fontweight='bold', pad=15)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.hist(offensive_confs, bins=5, color='#e74c3c', alpha=0.7, edgecolor='black')
if offensive_confs:
    ax2.axvline(x=np.mean(offensive_confs), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(offensive_confs):.1f}%')
ax2.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax2.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
ax2.set_title(f'Offensive Distribution (n={len(offensive_confs)})', fontsize=13, fontweight='bold', pad=15)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'logreg_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'logreg_distribution.png'}")
plt.close()

# Figure 3: Summary statistics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Pie chart - Classification results
labels = ['Not Offensive', 'Offensive']
sizes = [logreg_data['not_offensive_count'], logreg_data['offensive_count']]
colors_pie = ['#2ecc71', '#e74c3c']
explode = (0.05, 0.1)
ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Classification Results', fontsize=13, fontweight='bold', pad=15)

# Box plot - Confidence distribution by category
data_to_plot = [not_offensive_confs, offensive_confs]
bp = ax2.boxplot(data_to_plot, vert=True, patch_artist=True, labels=['Not Offensive', 'Offensive'])
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], linewidth=1.5)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax2.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_ylabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Confidence by Classification', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)

# Statistics text
stats_text = f"""
Model: {logreg_data['model']}

Total Messages: {logreg_data['total_messages']}
Offensive: {logreg_data['offensive_count']} ({logreg_data['offensive_count']/logreg_data['total_messages']*100:.1f}%)
Not Offensive: {logreg_data['not_offensive_count']} ({logreg_data['not_offensive_count']/logreg_data['total_messages']*100:.1f}%)

Confidence Statistics:
  Mean: {np.mean(confidences):.2f}%
  Median: {np.median(confidences):.2f}%
  Min: {np.min(confidences):.2f}%
  Max: {np.max(confidences):.2f}%
  Std Dev: {np.std(confidences):.2f}%

Offensive Messages Detected:
  • "you are ugly" (75.1%)
  • "women ☕" (53.7%)
  • "i was sick and my gf..." (58.6%)

✅ Conservative: Low false positive rate
"""
ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.3))
ax3.axis('off')
ax3.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=15)

# Offensive probability distribution
ax4.scatter(range(len(prob_offensive)), prob_offensive, 
           c=['#e74c3c' if o else '#2ecc71' for o in is_offensive],
           s=100, alpha=0.6, edgecolor='black', linewidth=1)
ax4.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5, label='50% Threshold')
ax4.set_xlabel('Message Index', fontsize=12, fontweight='bold')
ax4.set_ylabel('Offensive Probability (%)', fontsize=12, fontweight='bold')
ax4.set_title('Offensive Probability for Each Message', fontsize=13, fontweight='bold', pad=15)
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 100)

plt.suptitle('Logistic Regression Classifier Analysis - Conservative & Reliable', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(img_dir / 'logreg_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'logreg_summary.png'}")
plt.close()

# Figure 4: Comparison of offensive probability for edge cases
fig, ax = plt.subplots(figsize=(16, 8))

# Sort by offensive probability
sorted_indices = np.argsort(prob_offensive)[::-1]
sorted_messages = [messages[i] for i in sorted_indices]
sorted_probs = [prob_offensive[i] for i in sorted_indices]
sorted_offensive = [is_offensive[i] for i in sorted_indices]

colors = ['#e74c3c' if o else '#2ecc71' for o in sorted_offensive]
bars = ax.barh(range(len(sorted_messages)), sorted_probs, color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Decision Threshold')
ax.set_yticks(range(len(sorted_messages)))
ax.set_yticklabels(sorted_messages, fontsize=9)
ax.set_xlabel('Offensive Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('All Messages Ranked by Offensive Probability', 
             fontsize=14, fontweight='bold', pad=20)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label='Classified as Offensive'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label='Classified as Not Offensive'),
                   plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='50% Threshold')]
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, 100)

# Add value labels
for i, prob in enumerate(sorted_probs):
    ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(img_dir / 'logreg_ranked_probability.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'logreg_ranked_probability.png'}")
plt.close()

# Figure 5: Close calls (messages near the threshold)
close_calls_indices = [i for i, p in enumerate(prob_offensive) if 40 <= p <= 60]
if close_calls_indices:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    close_messages = [messages[i] for i in close_calls_indices]
    close_probs = [prob_offensive[i] for i in close_calls_indices]
    close_offensive = [is_offensive[i] for i in close_calls_indices]
    
    colors = ['#e74c3c' if o else '#2ecc71' for o in close_offensive]
    bars = ax.barh(range(len(close_messages)), close_probs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=50, color='black', linestyle='--', linewidth=3, label='50% Decision Threshold')
    ax.axvspan(45, 55, alpha=0.2, color='yellow', label='Uncertainty Zone (±5%)')
    
    ax.set_yticks(range(len(close_messages)))
    ax.set_yticklabels(close_messages, fontsize=10)
    ax.set_xlabel('Offensive Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Close Calls - Messages Near Decision Boundary (n={len(close_calls_indices)})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.set_xlim(35, 65)
    
    for i, prob in enumerate(close_probs):
        ax.text(prob + 0.5, i, f'{prob:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(img_dir / 'logreg_close_calls.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {img_dir / 'logreg_close_calls.png'}")
    plt.close()

print(f"\n🎨 All LogReg visualizations saved to {img_dir}/")
