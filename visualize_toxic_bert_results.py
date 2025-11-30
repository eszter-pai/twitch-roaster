import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load Toxic-BERT results
with open('test_classifiers/results/toxic_bert_results.json', 'r', encoding='utf-8') as f:
    toxic_bert_data = json.load(f)

# Create img directory if it doesn't exist
img_dir = Path('img')
img_dir.mkdir(exist_ok=True)

# Extract data
messages = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in toxic_bert_data['results']]
full_messages = [r['text'] for r in toxic_bert_data['results']]
categories = list(toxic_bert_data['label_mapping'].values())

# Create score matrix
score_matrix = []
for result in toxic_bert_data['results']:
    scores = [result['scores'][cat] * 100 for cat in categories]
    score_matrix.append(scores)

score_matrix = np.array(score_matrix)

# Figure 1: Heatmap of all toxicity scores
fig, ax = plt.subplots(figsize=(12, 16))
im = ax.imshow(score_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10, fontweight='bold')
ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=8)
ax.set_xlabel('Toxicity Categories', fontsize=12, fontweight='bold')
ax.set_ylabel('Messages', fontsize=12, fontweight='bold')
ax.set_title('Toxic-BERT: Toxicity Scores Heatmap\nAll 28 Messages Flagged (>50% in at least one category)', 
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score (%)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations for high scores
for i in range(len(messages)):
    for j in range(len(categories)):
        score = score_matrix[i, j]
        if score > 50:
            text = ax.text(j, i, f'{score:.0f}', ha="center", va="center",
                          color="white" if score > 70 else "black", fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig(img_dir / 'toxic_bert_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'toxic_bert_heatmap.png'}")
plt.close()

# Figure 2: Distribution of scores per category
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, category in enumerate(categories):
    scores = score_matrix[:, idx]
    ax = axes[idx]
    
    ax.hist(scores, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(scores), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(scores):.1f}%')
    ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
    ax.set_xlabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'{category.upper()}\n(High: {np.sum(scores > 50)}/28)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, 100)

plt.suptitle('Toxic-BERT: Score Distribution by Category', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(img_dir / 'toxic_bert_category_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'toxic_bert_category_distribution.png'}")
plt.close()

# Figure 3: Maximum toxicity score per message
max_scores = np.max(score_matrix, axis=1)
max_categories = [categories[np.argmax(score_matrix[i])] for i in range(len(messages))]

fig, ax = plt.subplots(figsize=(16, 10))
colors = ['#c0392b' if s > 80 else '#e74c3c' if s > 60 else '#f39c12' for s in max_scores]
bars = ax.barh(range(len(messages)), max_scores, color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=50, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=9)
ax.set_xlabel('Maximum Toxicity Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Toxic-BERT: Maximum Toxicity Score Per Message', 
             fontsize=14, fontweight='bold', pad=20)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label='Offensive (28)'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label='Not Offensive (0)'),
                   plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='50% Threshold')]
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, 100)

# Add value labels and category
for i, (bar, score, cat) in enumerate(zip(bars, max_scores, max_categories)):
    ax.text(score + 1, i, f'{score:.1f}% ({cat})', va='center', fontsize=7)

plt.tight_layout()
plt.savefig(img_dir / 'toxic_bert_max_scores.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'toxic_bert_max_scores.png'}")
plt.close()

# Figure 4: Average scores per category
avg_scores = np.mean(score_matrix, axis=0)
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(categories, avg_scores, color=['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#e8daef'],
              alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax.set_ylabel('Average Score (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_title('Toxic-BERT: Average Score by Category', fontsize=14, fontweight='bold', pad=20)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

# Add value labels on bars
for bar, score in zip(bars, avg_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{score:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(img_dir / 'toxic_bert_category_averages.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'toxic_bert_category_averages.png'}")
plt.close()

# Figure 5: Top 10 most toxic messages with breakdown
sorted_indices = np.argsort(max_scores)[::-1][:10]
top_messages = [messages[i] for i in sorted_indices]
top_scores_matrix = score_matrix[sorted_indices]

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(top_messages))
width = 0.13

for i, category in enumerate(categories):
    scores = top_scores_matrix[:, i]
    offset = width * (i - len(categories)/2 + 0.5)
    ax.bar(x + offset, scores, width, label=category, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Messages', fontsize=12, fontweight='bold')
ax.set_title('Toxic-BERT: Top 10 Most Toxic Messages - Category Breakdown', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(top_messages, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper right', ncol=2)
ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(img_dir / 'toxic_bert_top_toxic.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'toxic_bert_top_toxic.png'}")
plt.close()

# Figure 6: Summary dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top left: Overall statistics
ax1 = fig.add_subplot(gs[0, 0])
stats_text = f"""
Model: {toxic_bert_data['model']}

Total Messages: {toxic_bert_data['total_messages']}
High Toxicity (>50%): {toxic_bert_data['high_toxicity_count']}
False Positive Rate: EXTREMELY HIGH

Category Averages:
  • Toxic: {avg_scores[0]:.1f}%
  • Severe Toxic: {avg_scores[1]:.1f}%
  • Obscene: {avg_scores[2]:.1f}%
  • Threat: {avg_scores[3]:.1f}%
  • Insult: {avg_scores[4]:.1f}%
  • Identity Hate: {avg_scores[5]:.1f}%

Max Score: {np.max(max_scores):.1f}%
Min Score: {np.min(max_scores):.1f}%
Mean Score: {np.mean(max_scores):.1f}%

⚠️ Even benign messages like
"hello hows your day" and
"nice play!" flagged as toxic!
"""
ax1.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='#ffcccc', alpha=0.5))
ax1.axis('off')
ax1.set_title('Overall Statistics', fontsize=13, fontweight='bold', pad=15)

# Top right: Pie chart of high scores
ax2 = fig.add_subplot(gs[0, 1])
high_score_counts = [np.sum(score_matrix[:, i] > 50) for i in range(len(categories))]
colors_pie = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#e8daef']
ax2.pie(high_score_counts, labels=categories, colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title(f'Messages with >50% Score by Category\n(Total: {toxic_bert_data["high_toxicity_count"]} messages)', 
              fontsize=12, fontweight='bold', pad=15)

# Middle: Box plot comparison
ax3 = fig.add_subplot(gs[1, :])
bp = ax3.boxplot([score_matrix[:, i] for i in range(len(categories))], 
                  patch_artist=True, labels=categories)
for patch, color in zip(bp['boxes'], colors_pie):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], linewidth=1.5)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax3.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5, label='50% Threshold')
ax3.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Score Distribution by Category (Box Plot)', fontsize=13, fontweight='bold', pad=15)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 100)

# Bottom: Histogram of max scores
ax4 = fig.add_subplot(gs[2, :])
ax4.hist(max_scores, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axvline(x=np.mean(max_scores), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(max_scores):.1f}%')
ax4.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax4.set_xlabel('Maximum Toxicity Score (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of Maximum Toxicity Scores', fontsize=13, fontweight='bold', pad=15)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_xlim(0, 100)

plt.suptitle('Toxic-BERT Comprehensive Analysis - High False Positive Rate', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(img_dir / 'toxic_bert_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'toxic_bert_summary.png'}")
plt.close()

print(f"\n🎨 All Toxic-BERT visualizations saved to {img_dir}/")
print(f"\n📊 Key Findings:")
print(f"  • All {toxic_bert_data['total_messages']} messages flagged as toxic (>50%)")
print(f"  • 'Toxic' category dominates with {avg_scores[0]:.1f}% average")
print(f"  • Highest score: {np.max(max_scores):.1f}% ({full_messages[np.argmax(max_scores)][:50]}...)")
print(f"  • Even benign messages heavily flagged (e.g., 'hello hows your day': {score_matrix[15, 0]:.1f}%)")
