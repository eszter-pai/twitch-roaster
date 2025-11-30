import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load Combined (ToxicBERT + Zero-shot) results at 85% threshold
results_path = Path(__file__).parent / 'test_classifiers' / 'results' / 'toxicbert_zeroshot_results_threshold_85.json'
with open(results_path, 'r', encoding='utf-8') as f:
    combined_data = json.load(f)

# Create img directory if it doesn't exist
img_dir = Path('img')
img_dir.mkdir(exist_ok=True)

# Extract data
messages = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in combined_data['results']]
full_messages = [r['text'] for r in combined_data['results']]
max_scores = [r['max_score'] * 100 for r in combined_data['results']]
is_toxic = [r['is_toxic'] for r in combined_data['results']]
model_used = [r['model_used'] for r in combined_data['results']]
toxic_bert_scores = [r['toxic_bert_score'] * 100 for r in combined_data['results']]
zero_shot_scores = [r['zero_shot_score'] * 100 for r in combined_data['results']]
max_labels = [r['max_label'].replace('chat message contains ', '') if 'chat message' in r['max_label'] else r['max_label'] 
              for r in combined_data['results']]

# Figure 1: Combined classifier results
fig, ax = plt.subplots(figsize=(16, 10))
colors = ['#e74c3c' if t else '#2ecc71' for t in is_toxic]
bars = ax.barh(range(len(messages)), max_scores, color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=85, color='black', linestyle='--', linewidth=2, label='85% Threshold', alpha=0.7)
ax.set_yticks(range(len(messages)))
ax.set_yticklabels(messages, fontsize=9)
ax.set_xlabel('Max Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Combined Classifier (ToxicBERT + Zero-Shot) @ 85% Threshold', 
             fontsize=14, fontweight='bold', pad=20)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.7, label=f'Offensive ({combined_data["toxic_count"]}'),
                   Patch(facecolor='#2ecc71', alpha=0.7, label=f'Not Offensive ({combined_data["ok_count"]}'),
                   plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='85% Threshold')]
ax.legend(handles=legend_elements, loc='lower right')
ax.set_xlim(0, 100)

# Add value labels
for i, (score, label, model) in enumerate(zip(max_scores, max_labels, model_used)):
    model_short = 'TB' if model == 'toxic-bert' else 'ZS'
    ax.text(score + 1, i, f'{score:.1f}% ({model_short})', va='center', fontsize=7)

plt.tight_layout()
plt.savefig(img_dir / 'combined_85_results.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'combined_85_results.png'}")
plt.close()

# Figure 2: Model usage comparison (which model "won" more often)
model_counts = {'toxic-bert': model_used.count('toxic-bert'), 
                'zero-shot': model_used.count('zero-shot')}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors_pie = ['#e74c3c', '#3498db']
ax1.pie(model_counts.values(), labels=model_counts.keys(), colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Which Model Had Higher Score', fontsize=13, fontweight='bold', pad=15)

# Bar chart
ax2.bar(model_counts.keys(), model_counts.values(), color=colors_pie, alpha=0.7, 
        edgecolor='black', linewidth=2)
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Model Usage Frequency', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)
for i, (k, v) in enumerate(model_counts.items()):
    ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(img_dir / 'combined_85_model_usage.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'combined_85_model_usage.png'}")
plt.close()

# Figure 3: Score comparison - ToxicBERT vs Zero-Shot
fig, ax = plt.subplots(figsize=(12, 12))

# Scatter plot
colors_scatter = ['#e74c3c' if t else '#2ecc71' for t in is_toxic]
ax.scatter(toxic_bert_scores, zero_shot_scores, c=colors_scatter, s=100, 
           alpha=0.6, edgecolor='black', linewidth=1.5)

# Add diagonal line (where both models agree)
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=2, label='Equal scores')

# Add threshold lines
ax.axvline(x=85, color='red', linestyle='--', linewidth=2, alpha=0.5, label='85% Threshold')
ax.axhline(y=85, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Add quadrant labels
ax.text(95, 95, 'Both High\n(Definitely Toxic)', ha='right', va='top', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
ax.text(5, 5, 'Both Low\n(Definitely OK)', ha='left', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))
ax.text(95, 5, 'TB High, ZS Low\n(TB Triggers)', ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ffe6cc', alpha=0.7))
ax.text(5, 95, 'ZS High, TB Low\n(ZS Triggers)', ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#cce6ff', alpha=0.7))

ax.set_xlabel('Toxic-BERT Score (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Zero-Shot Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Score Comparison: Toxic-BERT vs Zero-Shot\n(Taking Maximum at 85% Threshold)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='center', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(img_dir / 'combined_85_score_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'combined_85_score_comparison.png'}")
plt.close()

# Figure 4: Score differences
score_differences = [tb - zs for tb, zs in zip(toxic_bert_scores, zero_shot_scores)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Histogram of differences
ax1.hist(score_differences, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No difference')
ax1.axvline(x=np.mean(score_differences), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(score_differences):.1f}%')
ax1.set_xlabel('Score Difference (Toxic-BERT - Zero-Shot) %', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Score Differences Between Models', fontsize=13, fontweight='bold', pad=15)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Bar chart showing which model scored higher for each message
ax2.barh(range(len(messages)), score_differences, 
        color=['#e74c3c' if d > 0 else '#3498db' for d in score_differences],
        alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax2.set_yticks(range(len(messages)))
ax2.set_yticklabels(messages, fontsize=8)
ax2.set_xlabel('Score Difference (%)\n← Zero-Shot Higher | Toxic-BERT Higher →', 
               fontsize=11, fontweight='bold')
ax2.set_title('Per-Message Score Differences', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'combined_85_score_differences.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'combined_85_score_differences.png'}")
plt.close()

# Figure 5: Summary dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top left: Statistics
ax1 = fig.add_subplot(gs[0, 0])
stats_text = f"""
Combined Classifier (Max Score Strategy)
Threshold: {combined_data['threshold_percentage']}

Total Messages: {combined_data['total_messages']}
Toxic: {combined_data['toxic_count']} ({combined_data['toxic_count']/combined_data['total_messages']*100:.1f}%)
OK: {combined_data['ok_count']} ({combined_data['ok_count']/combined_data['total_messages']*100:.1f}%)

Model Usage:
  • Toxic-BERT: {model_counts['toxic-bert']}/28
  • Zero-Shot: {model_counts['zero-shot']}/28

Score Statistics:
  Max Score Range: {np.min(max_scores):.1f}% - {np.max(max_scores):.1f}%
  Mean Max Score: {np.mean(max_scores):.1f}%

Difference Stats:
  Mean Diff (TB-ZS): {np.mean(score_differences):.1f}%
  TB Often Higher: {sum(1 for d in score_differences if d > 0)}/28

Messages Caught: 22/28 (78.6%)
False Negatives: 6 messages below 85%
"""
ax1.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='#e8f4f8', alpha=0.5))
ax1.axis('off')
ax1.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=15)

# Top right: Pie chart
ax2 = fig.add_subplot(gs[0, 1])
labels = ['Toxic', 'OK']
sizes = [combined_data['toxic_count'], combined_data['ok_count']]
colors_pie2 = ['#e74c3c', '#2ecc71']
explode = (0.05, 0.1)
ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie2, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title(f'Classification @ {combined_data["threshold_percentage"]} Threshold', 
              fontsize=13, fontweight='bold', pad=15)

# Middle left: Box plot comparison
ax3 = fig.add_subplot(gs[1, 0])
toxic_max_scores = [s for s, t in zip(max_scores, is_toxic) if t]
ok_max_scores = [s for s, t in zip(max_scores, is_toxic) if not t]
bp = ax3.boxplot([toxic_max_scores, ok_max_scores], patch_artist=True, 
                  tick_labels=['Toxic', 'OK'])
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#2ecc71')
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], linewidth=1.5)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax3.axhline(y=85, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_ylabel('Max Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Score Distribution by Classification', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)

# Middle right: Model contribution
ax4 = fig.add_subplot(gs[1, 1])
toxic_model_usage = {'toxic-bert': 0, 'zero-shot': 0}
ok_model_usage = {'toxic-bert': 0, 'zero-shot': 0}
for t, m in zip(is_toxic, model_used):
    if t:
        toxic_model_usage[m] += 1
    else:
        ok_model_usage[m] += 1

x = np.arange(2)
width = 0.35
bars1 = ax4.bar(x - width/2, [toxic_model_usage['toxic-bert'], toxic_model_usage['zero-shot']], 
                width, label='Toxic', color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax4.bar(x + width/2, [ok_model_usage['toxic-bert'], ok_model_usage['zero-shot']], 
                width, label='OK', color='#2ecc71', alpha=0.7, edgecolor='black')
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Which Model Dominated by Classification', fontsize=13, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(['Toxic-BERT', 'Zero-Shot'])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Bottom: Score comparison scatter (simplified)
ax5 = fig.add_subplot(gs[2, :])
for i, (tb, zs, toxic, msg) in enumerate(zip(toxic_bert_scores, zero_shot_scores, is_toxic, messages)):
    color = '#e74c3c' if toxic else '#2ecc71'
    ax5.scatter(tb, zs, c=color, s=80, alpha=0.6, edgecolor='black', linewidth=1)
ax5.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1.5, label='Equal scores')
ax5.axvline(x=85, color='red', linestyle='--', linewidth=2, alpha=0.4)
ax5.axhline(y=85, color='red', linestyle='--', linewidth=2, alpha=0.4)
ax5.set_xlabel('Toxic-BERT Score (%)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Zero-Shot Score (%)', fontsize=12, fontweight='bold')
ax5.set_title('Model Score Comparison (Red/Green = Toxic/OK)', fontsize=13, fontweight='bold', pad=15)
ax5.legend()
ax5.grid(alpha=0.3)
ax5.set_xlim(0, 100)
ax5.set_ylim(0, 100)

plt.suptitle('Combined Classifier Analysis (ToxicBERT + Zero-Shot @ 85% Threshold)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(img_dir / 'combined_85_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {img_dir / 'combined_85_summary.png'}")
plt.close()

# Figure 6: Edge cases (messages close to threshold)
edge_case_indices = [i for i, s in enumerate(max_scores) if 75 <= s <= 95]
if edge_case_indices:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    edge_messages = [messages[i] for i in edge_case_indices]
    edge_max_scores = [max_scores[i] for i in edge_case_indices]
    edge_tb_scores = [toxic_bert_scores[i] for i in edge_case_indices]
    edge_zs_scores = [zero_shot_scores[i] for i in edge_case_indices]
    edge_toxic = [is_toxic[i] for i in edge_case_indices]
    
    x = np.arange(len(edge_messages))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, edge_tb_scores, width, label='Toxic-BERT', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, edge_zs_scores, width, label='Zero-Shot', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    
    ax.axhline(y=85, color='black', linestyle='--', linewidth=2, label='85% Threshold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Edge Cases: Messages Near 85% Threshold (n={len(edge_case_indices)})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(edge_messages, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add markers for final classification
    for i, toxic in enumerate(edge_toxic):
        marker = '✓' if toxic else '✗'
        color = '#e74c3c' if toxic else '#2ecc71'
        ax.text(i, max(edge_tb_scores[i], edge_zs_scores[i]) + 3, marker,
               ha='center', fontsize=16, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(img_dir / 'combined_85_edge_cases.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {img_dir / 'combined_85_edge_cases.png'}")
    plt.close()

print(f"\n🎨 All Combined Classifier (85%) visualizations saved to {img_dir}/")
print(f"\n📊 Key Findings:")
print(f"  • {combined_data['toxic_count']}/28 messages flagged as toxic (78.6%)")
print(f"  • 6 messages passed (below 85% threshold)")
print(f"  • Toxic-BERT dominated {model_counts['toxic-bert']} times, Zero-Shot {model_counts['zero-shot']} times")
print(f"  • Mean score difference (TB-ZS): {np.mean(score_differences):.1f}%")
print(f"  • Strategy: Taking MAX of both models makes it more sensitive overall")
