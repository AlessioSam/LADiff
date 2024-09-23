import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pylab import rcParams
import os

params = {
    'axes.labelsize': 25,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'figure.figsize': [1.7*4*1.5, 1.7*3*1.5]
}
rcParams.update(params)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rc('pdf', fonttype=42)
matplotlib.font_manager._load_fontmanager(try_read_cache=False)


# Initialize the dataframe with the specified columns and values taken from experiments
df = pd.DataFrame([['LADiff', 1, 0.31386853830657735, 0.291, 0.462, 0.583, 4.113],
                   ['LADiff', 2, 0.3330233057744085, 0.482, 0.690, 0.793, 0.147],
                   ['LADiff', 3, 0.35582907312417744, 0.483, 0.678, 0.781, 0.248],
                   ['LADiff', 4, 0.37182213979608875, 0.474, 0.674, 0.776, 0.255],
                   ['LADiff', 5, 0.37886515620768496, 0.451, 0.637, 0.737, 0.393],
                   ['MLD', 1, 0.3290426939664966, 0.253, 0.395, 0.491, 9.688],
                   ['MLD', 2, 0.33930744423480134, 0.457, 0.664, 0.769, 0.335],
                   ['MLD', 3, 0.35122589023532786, 0.459, 0.653, 0.757, 0.416],
                   ['MLD', 4, 0.3615048694049611, 0.439, 0.635, 0.750, 0.612],
                   ['MLD', 5, 0.3695599597072375, 0.402, 0.583, 0.689, 1.005],
                   ['T2M-GPT', 1, 0.3839793112693411, 0.313, 0.496, 0.611, 2.980],
                   ['T2M-GPT', 2, 0.4390971567481756, 0.464, 0.667, 0.772, 0.128],
                   ['T2M-GPT', 3, 0.5964586652558426, 0.469, 0.671, 0.772, 0.205],
                   ['T2M-GPT', 4, 0.6319396429910109, 0.483, 0.679, 0.775, 0.355],
                   ['T2M-GPT', 5, 0.7087907024792263, 0.417, 0.599, 0.702, 0.610],
                   ],
                  columns=['model', 'latents', 'time', 'top1', 'top2', 'top3', 'fid'])

def plot1():
    # Define colors, markers and convertion for the legend
    colors = {'LADiff': 'b', 'MLD': 'g', 'T2M-GPT': 'r'}
    markers = {1: 'o', 2: 'P', 3: '^', 4: 'D', 5: '*'}
    convertion = {1: '1-48 frames', 2: '49-96 frames', 3: '97-144 frames', 4: '145-192 frames', 5: '192+ frames'}

    # Create the scatter plot
    color_handles, color_labels = [], []
    marker_handles, marker_labels = [], []

    for key, group in df.groupby(['model', 'latents']):
        model, latents = key
        color = colors[model]
        marker = markers[latents]
        
        plt.scatter(group['time'], group['top3'], c=color, marker=marker, s=110,
                    label=f'{model} - {latents}')
        
        # Store color and marker legends separately
        if model not in color_labels:
            color_handles.append(mlines.Line2D([], [], marker="s", color=color, linestyle='None', markersize=8))
            color_labels.append(model)
        if convertion[latents] not in marker_labels:
            marker_handles.append(mlines.Line2D([], [], marker=marker, color='k', linestyle='None', markersize=8))
            marker_labels.append(convertion[latents])

    for model, color in colors.items():
        model_data = df[df['model'] == model]
        plt.plot(model_data['time'], model_data['top3'], c=color, linestyle='--', linewidth=1)

    # Customize the plot
    plt.xlabel('Average Inference Time per Sentence (AITS) in seconds')
    plt.ylabel('R-precision top3')

    # Create separate legends for colors and markers
    plt.legend(color_handles + [mlines.Line2D([], [], linestyle='None')]*2 + marker_handles,
                            color_labels + [""]*2 + marker_labels, title='Legend', ncols=2)

    # Save the plot to a file (in PNG format)
    plt.savefig('timevsperformance.png', bbox_inches='tight')

    # Show the plot (optional)
    plt.close()
    print(f'Plot saved to {os.getcwd()}/timevsperformance.png')


def plot2():
    # Separate the data for each model
    models = df['model'].unique()
    conversion = {1: '1-48', 2: '49-96', 3: '97-144', 4: '145-192', 5: '192+'}

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot lines for each model
    for model in models:
        model_data = df[df['model'] == model]
        ax.plot(model_data['latents'], model_data['time'], label=model, linewidth=2, marker='o')

    # Add labels and a legend
    ax.set_xlabel('Frames')
    ax.set_ylabel('Average inference time in seconds')
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(list(conversion.values()))
    ax.legend()

    # Save the plot to a file (in PNG format)
    plt.savefig('timevsperformance.png', bbox_inches='tight')

    # Show the plot (optional)
    plt.close()
    print(f'Plot saved to {os.getcwd()}/timevsperformance.png')


def plot3():
    # Separate the data for each model
    models = df['model'].unique()
    styles = {"MLD": "--", "LADiff": "-", "T2M-GPT": ":"}
    markers = {"MLD": "s", "LADiff": "o", "T2M-GPT": "^"}

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot lines for each model on the left y-axis
    for model in models:
        model_data = df[df['model'] == model]
        ax1.plot(model_data['latents'], model_data['time'], color='tab:blue', linestyle=styles[model], marker=markers[model])

    # Add labels for the left y-axis
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Average inference time in seconds', color='tab:blue')
    ax1.set_xticks(range(1, 6))
    ax1.set_xticklabels(['1-48', '49-96', '97-144', '145-192', '192+'])

    # Create a second y-axis on the right side
    ax2 = ax1.twinx()

    # Plot lines for 'top3' on the right y-axis
    for model in models:
        model_data = df[df['model'] == model]
        ax2.plot(model_data['latents'], model_data['top3'], color='tab:red', linestyle=styles[model], marker=markers[model])

    # Add a legend for the right y-axis
    ax2.set_ylabel('R-precision top 3', color='tab:red')

    # Combine legends from both y-axes
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    # Create separate legends for colors and markers
    plt.legend([mlines.Line2D([], [], color='k', linestyle=styles["LADiff"], marker=markers["LADiff"]),
                mlines.Line2D([], [], color='k', linestyle=styles["MLD"], marker=markers["MLD"]),
                mlines.Line2D([], [], color='k', linestyle=styles["T2M-GPT"], marker=markers["T2M-GPT"]),
                mlines.Line2D([], [], color="tab:blue"),
                mlines.Line2D([], [], color="tab:red"),
                ],
                ["LADiff", "MLD", "T2M-GPT", "Time", "R-precision"], loc=(0.69, 0.23))

    # Save the plot to a file (in PNG format)
    plt.savefig('timevsperformance.png',bbox_inches='tight')

    # Show the plot (optional)
    plt.close()
    print(f'Plot saved to {os.getcwd()}/timevsperformance.png')

plot3()
