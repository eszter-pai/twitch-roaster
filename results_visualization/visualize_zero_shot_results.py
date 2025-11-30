import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load Zero-shot results
with open('test_classifiers/results/zero_shot_results.json', 'r', encoding='utf-8') as f:
    zero_shot_data = json.load(f)

# Create img directory if it doesn't exist
img_dir = Path('img')
img_dir.mkdir(exist_ok=True)

# Extract data
messages = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in zero_shot_data['results']]
max_scores = [r['max_score'] * 100 for r in zero_shot_data['results']]
is_offensive = [r['is_likely_offensive'] for r in zero_shot_data['results']]
categories = zero_shot_data['offensive_categories']

# Get top category for each message
top_categories = []
for result in zero_shot_data['results']:
    cats = result['categories']
    top_cat = max(cats, key=cats.get)
    # Shorten category name
    short_cat = top_cat.replace('chat message contains ', '')
    top_categories.append(short_cat)

# Figure 1: Max scores bar chart
fig, ax = plt.subplots(figsize=(16, 10))
colors = ['#e74c3c' if o else '#2ecc71' for o in is_offensive]
bars = ax.barh(range(len(messages)), max_scores, color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold', alpha=0.7)
ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=9)
ax.set_xlabel('Max Confidence Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Zero-Shot Classifier Results - High False Positive Rate', 
             fontsize=14, fontweight='bold', pad=20)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label=f'Offensive ({zero_shot_data["offensive_count"]})'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label=f'Not Offensive ({zero_shot_data["not_offensive_count"]})'),
                   plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='50% Threshold')]
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, 100)

# Add value labels
for i, (bar, score, cat) in enumerate(zip(bars, max_scores, top_categories)):
    ax.text(score + 1, i, f'{score:.1f}% ({cat})', va='center', fontsize=7)

plt.tight_layout()
plt.savefig(img_dir / 'zero_shot_max_scores.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'zero_shot_max_scores.png'}")
plt.close()

# Figure 2: Score distribution by classification
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

offensive_scores = [s for s, o in zip(max_scores, is_offensive) if o]
not_offensive_scores = [s for s, o in zip(max_scores, is_offensive) if not o]

ax1.hist(offensive_scores, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.axvline(x=np.mean(offensive_scores), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(offensive_scores):.1f}%')
ax1.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax1.set_xlabel('Max Score (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title(f'Offensive Distribution (n={len(offensive_scores)})', fontsize=13, fontweight='bold', pad=15)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.hist(not_offensive_scores, bins=10, color='#2ecc71', alpha=0.7, edgecolor='black')
if not_offensive_scores:
    ax2.axvline(x=np.mean(not_offensive_scores), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(not_offensive_scores):.1f}%')
ax2.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax2.set_xlabel('Max Score (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title(f'Not Offensive Distribution (n={len(not_offensive_scores)})', fontsize=13, fontweight='bold', pad=15)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'zero_shot_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'zero_shot_distribution.png'}")
plt.close()

# Figure 3: Category breakdown heatmap
score_matrix = []
for result in zero_shot_data['results']:
    scores = [result['categories'][cat] * 100 for cat in categories]
    score_matrix.append(scores)

score_matrix = np.array(score_matrix)
short_categories = [cat.replace('chat message contains ', '') for cat in categories]

fig, ax = plt.subplots(figsize=(12, 16))
im = ax.imshow(score_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(short_categories)))
ax.set_xticklabels(short_categories, rotation=45, ha='right', fontsize=10, fontweight='bold')
ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=8)
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Messages', fontsize=12, fontweight='bold')
ax.set_title('Zero-Shot: Category Scores Heatmap\n25/28 Messages Flagged as Offensive', 
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score (%)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations for high scores
for i in range(len(messages)):
    for j in range(len(short_categories)):
        score = score_matrix[i, j]
        if score > 50:
            text = ax.text(j, i, f'{score:.0f}', ha="center", va="center",
                          color="white" if score > 70 else "black", fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig(img_dir / 'zero_shot_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'zero_shot_heatmap.png'}")
plt.close()

# Figure 4: Category frequency analysis
category_counts = {}
for cat in short_categories:
    category_counts[cat] = 0

for top_cat in top_categories:
    category_counts[top_cat] = category_counts.get(top_cat, 0) + 1

fig, ax = plt.subplots(figsize=(12, 7))
colors_bar = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#3498db']
bars = ax.bar(category_counts.keys(), category_counts.values(), 
              color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_title('Zero-Shot: Which Category Scored Highest (Most Common Trigger)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, count in zip(bars, category_counts.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(img_dir / 'zero_shot_category_frequency.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'zero_shot_category_frequency.png'}")
plt.close()

# Figure 5: Average scores per category
avg_scores = np.mean(score_matrix, axis=0)
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(short_categories, avg_scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax.set_ylabel('Average Score (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_title('Zero-Shot: Average Score by Category', fontsize=14, fontweight='bold', pad=20)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

# Add value labels
for bar, score in zip(bars, avg_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{score:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(img_dir / 'zero_shot_category_averages.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'zero_shot_category_averages.png'}")
plt.close()

# Figure 6: Summary dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top left: Statistics
ax1 = fig.add_subplot(gs[0, 0])
stats_text = f"""
Model: {zero_shot_data['model']}
Type: {zero_shot_data['classifier_type']}

Total Messages: {zero_shot_data['total_messages']}
Offensive: {zero_shot_data['offensive_count']} ({zero_shot_data['offensive_count']/zero_shot_data['total_messages']*100:.1f}%)
Not Offensive: {zero_shot_data['not_offensive_count']} ({zero_shot_data['not_offensive_count']/zero_shot_data['total_messages']*100:.1f}%)

Threshold: {zero_shot_data['threshold']*100:.0f}%

Max Score Range: {np.min(max_scores):.1f}% - {np.max(max_scores):.1f}%
Mean Score: {np.mean(max_scores):.1f}%

Most Common Trigger:
  "{max(category_counts, key=category_counts.get)}"
  ({category_counts[max(category_counts, key=category_counts.get)]}/28 messages)

⚠️ Major Issue: "requests to add on 
social platforms" incorrectly triggers 
for normal chat like:
  • "hello hows your day" (70.1%)
  • "nice play!" (73.1%)
"""
ax1.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='#ffe6e6', alpha=0.5))
ax1.axis('off')
ax1.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=15)

# Top right: Pie chart
ax2 = fig.add_subplot(gs[0, 1])
labels = ['Offensive', 'Not Offensive']
sizes = [zero_shot_data['offensive_count'], zero_shot_data['not_offensive_count']]
colors_pie = ['#e74c3c', '#2ecc71']
explode = (0.05, 0.1)
ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Classification Results', fontsize=13, fontweight='bold', pad=15)

# Middle: Category trigger frequency
ax3 = fig.add_subplot(gs[1, :])
ax3.bar(category_counts.keys(), category_counts.values(), 
       color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('Which Category Had Highest Score (Primary Trigger)', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)
for i, (k, v) in enumerate(category_counts.items()):
    ax3.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Bottom left: Score distribution
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(max_scores, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axvline(x=np.mean(max_scores), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(max_scores):.1f}%')
ax4.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
ax4.set_xlabel('Max Score (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Max Score Distribution', fontsize=13, fontweight='bold', pad=15)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Bottom right: Box plot
ax5 = fig.add_subplot(gs[2, 1])
bp = ax5.boxplot([offensive_scores, not_offensive_scores], 
                  patch_artist=True, labels=['Offensive', 'Not Offensive'])
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#2ecc71')
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], linewidth=1.5)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax5.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax5.set_ylabel('Max Score (%)', fontsize=12, fontweight='bold')
ax5.set_title('Score Distribution by Classification', fontsize=13, fontweight='bold', pad=15)
ax5.grid(axis='y', alpha=0.3)

plt.suptitle('Zero-Shot Classifier Analysis - Problematic Category Triggers', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(img_dir / 'zero_shot_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'zero_shot_summary.png'}")
plt.close()

# Figure 7: False positive examples
false_positives = []
for i, (msg, full_msg, is_off, score) in enumerate(zip(messages, 
                                                         [r['text'] for r in zero_shot_data['results']], 
                                                         is_offensive, 
                                                         max_scores)):
    # Identify obvious false positives
    if is_off and any(keyword in full_msg.lower() for keyword in 
                      ['hello', 'nice', 'thanks', 'gg', 'good', 'build', 'play']):
        false_positives.append((msg, score, top_categories[i]))

if false_positives:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    fp_messages = [fp[0] for fp in false_positives]
    fp_scores = [fp[1] for fp in false_positives]
    fp_cats = [fp[2] for fp in false_positives]
    
    bars = ax.barh(range(len(fp_messages)), fp_scores, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
    ax.set_yticks(range(len(fp_messages)))
    ax.set_yticklabels(fp_messages, fontsize=10)
    ax.set_xlabel('Max Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Zero-Shot: Clear False Positives\n(Benign messages incorrectly flagged as offensive)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.set_xlim(0, 100)
    
    for i, (score, cat) in enumerate(zip(fp_scores, fp_cats)):
        ax.text(score + 1, i, f'{score:.1f}% ({cat})', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(img_dir / 'zero_shot_false_positives.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {img_dir / 'zero_shot_false_positives.png'}")
    plt.close()

print(f"\n🎨 All Zero-Shot visualizations saved to {img_dir}/")
print(f"\n📊 Key Findings:")
print(f"  • {zero_shot_data['offensive_count']}/28 messages flagged as offensive")
print(f"  • Major issue: '{max(category_counts, key=category_counts.get)}' triggers on normal chat")
print(f"  • False positives include: 'hello hows your day', 'nice play!'")
print(f"  • Average score: {np.mean(max_scores):.1f}%")
