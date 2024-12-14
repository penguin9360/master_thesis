import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import re

experiment_name = "50k"
experiment_mode = "multiclass" # regression or multiclass
hpo_result_path = f"./hpo/{experiment_name}/{experiment_mode}/logs/"
new_search_boundary_color = 'green'
old_search_boundary_color = 'blue'
chosen_values_color = 'orange'

def extract_date_from_path(path):
    date_pattern = r'(\d{4}_\d{4}_\d{2}_\d{2}_\d{2})'
    match = re.search(date_pattern, str(path))
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y_%m%d_%H_%M_%S')
    return None

def extract_test_metric(log_file):
    metrics = []
    with open(log_file) as f:
        for line in f:
            if 'Overall test' in line:
                value = float(line.split('=')[1].split('+/-')[0].strip())
                metrics.append(value)
    return metrics

def read_and_group_test_metrics(experiment_name, experiment_mode):
    base_path = Path(f'hpo/{experiment_name}/{experiment_mode}')
    model_path = base_path / 'models'
    
    pre_dec_metrics = []
    post_dec_metrics = []
    
    print("\nSearching for quiet.log files...")
    for model_dir in model_path.glob('*'):
        if not model_dir.is_dir():
            continue
            
        log_file = model_dir / 'quiet.log'
        if not log_file.exists():
            continue
            
        date = extract_date_from_path(model_dir)
        if not date:
            continue
            
        print(f"\nProcessing {log_file}:")
        print(f"Date: {date}")
        
        metrics = extract_test_metric(log_file)
        print(f"Found metrics: {metrics}")
        
        if date.month < 12:
            pre_dec_metrics.extend(metrics)
        else:
            post_dec_metrics.extend(metrics)
            
    print("\nMetrics grouping summary:")
    print(f"Pre-December metrics: {len(pre_dec_metrics)} values")
    print(f"Post-December metrics: {len(post_dec_metrics)} values")
    
    return {'pre_december': pre_dec_metrics, 'post_december': post_dec_metrics}

def read_hpo_result_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        
    # Parse first line more carefully
    meta = lines[0].strip()
    exp_name = meta.split(',')[0].split(':')[1].strip()  # Just get "50k"
    mode = meta.split(',')[1].split(':')[1].strip()
    num_evals = int(meta.split(',')[2].split(':')[1].strip())
    search_opt = meta.split(',')[3].split(':')[1].strip()
    
    # Extract parameter values - adjust range to include batch_size
    params = {}
    for line in lines[4:10]:  # Changed from 4:9 to 4:10
        if line.strip():
            try:
                key, value = line.strip().split(':')
                params[key.strip()] = float(value)
            except:
                continue
                
    print(f"Parsed exp_name: {exp_name}")  # Debug print
    print(f"Parsed params: {params}")  # Add debug print
            
    return {
        'exp_name': exp_name,
        'mode': mode, 
        'num_evals': num_evals,
        'search_opt': search_opt,
        'params': params
    }

def plot_hpo_result_layover_graphs(experiment_name, experiment_mode, search_option):
    test_metrics = read_and_group_test_metrics(experiment_name, experiment_mode)
    # print(f"\nTest metrics: {test_metrics}")
    result_folder = hpo_result_path
    figure_path = f"./figures/hpo/hpo_overlay_{experiment_name}_{experiment_mode}_{search_option}.png"
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    param_limits = {
        'epochs': (1, 260), # originally (10, 250), adjusted to include the margins
        'depth': (2, 10),  # orignally (3, 9), adjusted to include the margins
        'init_lr': (1e-6, 1.7e-4), # originally (1e-5, 1.5e-4), adjusted to include the margins
        'max_lr': (7.1e-4, 1.29e-3), # originally (7.5e-4, 1.25e-3), adjusted to include the margins
        'batch_size': (41, 99), # originally (48, 96), adjusted to include the margins
        'test_metrics': (0, 3),  # Add appropriate range based on your metrics
    }
    if experiment_mode == 'multiclass':
        param_limits['test_metrics'] = (0.75, 0.95)
    else:
        param_limits['test_metrics'] = (2.25, 2.45)

    original_bounds = {
        'epochs': (10, 250),
        'depth': (3, 9),
        'init_lr': (1e-5, 1.5e-4),
        'max_lr': (7.5e-4, 1.25e-3),
        'batch_size': (48, 96)
    }

    ref_values = {
        'epochs': 150,
        'depth': 6,
        'init_lr': 0.0001,
        'max_lr': 0.001,
        'batch_size': 64
    }
    if experiment_mode == 'multiclass':
        ref_values['test_metrics'] = 0.857 # acutal value in thesis 
    else:
        ref_values['test_metrics'] = 2.35 # acutal value in thesis 

    print(f"Searching in: {result_folder}")
    files = [f for f in os.listdir(result_folder) if f.endswith('.txt')]
    print(f"Found {len(files)} txt files")
    
    result_groups = defaultdict(list)
    
    for file in files:
        result = read_hpo_result_file(os.path.join(result_folder, file))
        print(f"\nProcessing {file}:")
        print(f"Experiment: {result['exp_name']}")
        print(f"Parameters: {result['params']}")
        
        if result['exp_name'] != experiment_name:
            print(f"Skipping non-matching experiment")
            continue
            
        group_key = f"{result['mode']}_{result['search_opt']}_evals_{result['num_evals']}_parallel_runs"
        result_groups[group_key].append(result['params'])
    

    for group_key in result_groups.keys():
        print("====================== current key: ", group_key, "===================")
        if 'evals_10' in group_key:
            result_groups[group_key].append({'test_metrics': test_metrics['pre_december']})
            print(result_groups[group_key])
        else:
            result_groups[group_key].append({'test_metrics': test_metrics['post_december']})
            print(result_groups[group_key])
    
    

    print(f"\nFound {len(result_groups)} groups:")
    for group_key, results in result_groups.items():
        print(f"\nGroup {group_key}:")
        print(f"Number of results: {len(results)}")
        print(f"Sample values: {results[0]}")

    param_names = ['test_metrics', 'epochs', 'depth', 'init_lr', 'max_lr', 'batch_size']
    fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 15))
    colors = plt.cm.Set2(np.linspace(0, 1, len(result_groups)))
    
    for ax_idx, (param, ax) in enumerate(zip(param_names, axes)):
        for group_idx, (group_key, results) in enumerate(result_groups.items()):
            if param == 'test_metrics':
                for group_idx, (group_key, results) in enumerate(result_groups.items()):
                    metrics = next(r['test_metrics'] for r in results if 'test_metrics' in r)
                    # Create evenly spaced x coordinates within param limits
                    x_min, x_max = param_limits['test_metrics']
                    x = np.linspace(x_min, x_max, len(metrics))
                    ax.scatter(metrics, [0] * len(metrics), alpha=0.6, label=group_key, color=colors[group_idx])
                    ref_text = f'{ref_values["test_metrics"]:.3f}'
                    ax.text(ref_values["test_metrics"], 1.02, ref_text,
                            color=chosen_values_color, rotation=0,
                            ha='center', va='bottom', transform=ax.get_xaxis_transform())
                
                ax.grid(True)
                ax.set_xlim(param_limits['test_metrics'])
                ax.set_xlabel('Test metric value')
            if param == 'depth':
                group_counts = []
                for group_key, results in result_groups.items():
                    counts = np.zeros(7)  # depths 3-9
                    values = [r[param] for r in results if 'test_metrics' not in r]
                    for v in values:
                        counts[int(v)-3] += 1
                    group_counts.append(counts)
                
                #stacked bars
                bottom = np.zeros(7)
                for group_idx, counts in enumerate(group_counts):
                    normalized_counts = counts / np.max(np.sum(group_counts, axis=0)) * 0.8
                    ax.bar(range(3,10), normalized_counts, 
                        bottom=bottom,
                        alpha=0.6, 
                        color=colors[group_idx],
                        )
                    bottom += normalized_counts
                        
                total_counts = np.sum(group_counts, axis=0)
                max_count = int(np.max(total_counts))
                num_ticks = 5
                ytick_values = np.linspace(0, max_count * 1.2, num_ticks)
                ytick_positions = ytick_values / max_count * 0.8
                ax.set_yticks(ytick_positions)
                ax.set_yticklabels([f"{int(v)}" for v in ytick_values])
                ax.axvline(x=ref_values[param], color=chosen_values_color, linestyle='--',
                  label='Reference values' if ax_idx==0 else '')
                ref_text = f'{ref_values[param]:.0f}'
                ax.text(ref_values[param], 1.02, ref_text, color=chosen_values_color, rotation=0,
                    ha='center', va='bottom')
                ax.set_xticks(range(3,10))

            else:
                # For test_metrics, only get items that have it
                if param == 'test_metrics':
                    metrics = next(r['test_metrics'] for r in results if 'test_metrics' in r)
                    values = metrics
                else:
                    # For other params, skip the test_metrics dict
                    values = [r[param] for r in results if 'test_metrics' not in r]
                y_pos = [0] * len(values)
                ax.scatter(values, y_pos, alpha=0.6, 
                          label=group_key, color=colors[group_idx])
                ax.axvline(x=ref_values[param], color=chosen_values_color, linestyle='--',
                  label='Chosen values' if ax_idx==0 else '')
                if param != 'test_metrics': 
                    group_key = next(iter(result_groups))
                    ref_text = f'{ref_values[param]:.0f}' if ref_values[param] >= 1 else f'{ref_values[param]:.2e}'
                    ax.text(ref_values[param], 1.02, ref_text, 
                            color=chosen_values_color, rotation=0,
                            ha='center', va='bottom', transform=ax.get_xaxis_transform())
        
        if param != 'epochs' and param != 'test_metrics':
            ax.axvline(x=original_bounds[param][0], color=new_search_boundary_color, linestyle='-', 
                label='Search boundary') 
            ax.axvline(x=original_bounds[param][1], color=new_search_boundary_color, linestyle='-')
            lower_text = f'{original_bounds[param][0]:.0f}' if original_bounds[param][0] >= 1 else f'{original_bounds[param][0]:.2e}'
            upper_text = f'{original_bounds[param][1]:.0f}' if original_bounds[param][1] >= 1 else f'{original_bounds[param][1]:.2e}'
            
            # Use exact data coordinates for positioning
            ax.text(original_bounds[param][0], 1.02, lower_text,
                    color=new_search_boundary_color, rotation=0,
                    ha='center', va='bottom', transform=ax.get_xaxis_transform())
            ax.text(original_bounds[param][1], 1.02, upper_text,
                    color=new_search_boundary_color, rotation=0,
                    ha='center', va='bottom', transform=ax.get_xaxis_transform())
        else:
            if ax_idx > 0:  # Skip test_metrics subplot
                search_boundary_old_epoch = (50, 250)
                search_boundary_new_epoch = (10, 200)

                # Old boundaries with annotations
                ax.axvline(x=search_boundary_old_epoch[0], color=old_search_boundary_color, linestyle='-', 
                    label='Epoch search boundary for evals_10 parallel runs')
                ax.text(search_boundary_old_epoch[0], 1.02, str(search_boundary_old_epoch[0]), 
                        color=old_search_boundary_color, rotation=0, 
                        ha='center', va='bottom', transform=ax.get_xaxis_transform())
                
                ax.axvline(x=search_boundary_old_epoch[1], color=old_search_boundary_color, linestyle='-')
                ax.text(search_boundary_old_epoch[1], 1.02, str(search_boundary_old_epoch[1]), 
                        color=old_search_boundary_color, rotation=0, 
                        ha='center', va='bottom', transform=ax.get_xaxis_transform())

                # New boundaries with annotations  
                ax.axvline(x=search_boundary_new_epoch[0], color=new_search_boundary_color, linestyle='-', 
                    label='Search boundary' if (ax_idx==1 and group_idx==0) else '')
                ax.text(search_boundary_new_epoch[0], 1.02, str(search_boundary_new_epoch[0]), 
                        color=new_search_boundary_color, rotation=0,
                        ha='center', va='bottom', transform=ax.get_xaxis_transform())
                
                ax.axvline(x=search_boundary_new_epoch[1], color=new_search_boundary_color, linestyle='-')
                ax.text(search_boundary_new_epoch[1], 1.02, str(search_boundary_new_epoch[1]), 
                        color=new_search_boundary_color, rotation=0,
                        ha='center', va='bottom', transform=ax.get_xaxis_transform())
        
        ax.set_xlim(param_limits[param])
        ax.margins(x=0.02)
        ax.grid(True, axis='x')
        ax.set_xlabel(param)
        
        if param != 'depth':
            ax.set_yticks([])
    
    plt.tight_layout(pad=1.5)

    plt.tight_layout()
    
    plt.subplots_adjust(top=0.85)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
            loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=1)

    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    plt.close()

plot_hpo_result_layover_graphs(experiment_name, experiment_mode, 'random')
