import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys
import re

CUSTOM_PLOT_STYLE = {
    "text.usetex": True,

    "font.family": "serif",
    "font.size": 16,

    # Set some default sizes, so we don't need to include them in each call
    "lines.markersize": 1.5,
    "lines.linewidth": 1.5,
    # "errorbar.capsize": 3,

    # Set savefig to use bbox_inches="tight" by default
    "savefig.bbox": "tight",
    "savefig.facecolor": "none",
    "savefig.dpi": 300,

    # Higher DPI for sharper inline images
    "figure.dpi": 96
}

matplotlib.rcdefaults()
matplotlib.rcParams.update(CUSTOM_PLOT_STYLE)

# Define the regular expression pattern
pattern = r'nH[0-9\.e\+\-]+_lam([0-9\.e\+\-]+)um_theobs([0-9\.e\+\-]+)_Ldnu_xcentr\.txt'

# Function to extract values from filename
def extract_values(filename):
    match = re.match(pattern, filename)
    if match:
        lam = float(match.group(1))
        theobs = float(match.group(2))
        return lam, theobs
    else:
        return None, None

def main(filename):

    if filename.endswith('.txt'):
        lam, theobs = extract_values(filename)
        if lam is not None and theobs is not None:
            print(f'Filename: {filename}, lam: {lam}, theobs: {theobs}')
    
    with open(filename, 'r') as file:
        first_line = file.readline()
        first_row_data = first_line.split()[3:]
    
    # Assign the three floats to variables
    tobsmin, tobsmax, Ntobs = map(float, first_row_data)
    
    # Load the data from the text file
    data = np.genfromtxt(filename, dtype=float, skip_header=2, filling_values=np.nan)
    
    Ldnu_cgs = data[0]
    xcentr_pc = data[1]
    print("Max L = " + str(np.max(Ldnu_cgs)))

    log_tobsmin = np.log10(tobsmin)
    log_tobsmax = np.log10(tobsmax)
    
    # Create the logarithmic space with normalized values
    tobs = np.logspace(log_tobsmin, log_tobsmax, num=int(Ntobs), base=10)

    # Initialize the plot
    fig, ax1 = plt.subplots()
    
    # Plot Ldnu_cgs against tobs on the primary y-axis
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('$L_{\\nu}$ [cgs]')
    ax1.plot(tobs, Ldnu_cgs, color='tab:blue')
    ax1.tick_params(axis='y')
    
    # Create a secondary y-axis to plot xcentr_pc
    ax2 = ax1.twinx()
    ax2.set_ylabel('Flux centroid position [pc]')
    ax2.plot(tobs, xcentr_pc, color='tab:red')
    ax2.tick_params(axis='y')
    
    # Add a title and show the plot
    plt.title("$\\lambda = $"+str(lam)+"$\\mu$m, $\\theta_{obs}=$"+str(theobs))
    fig.tight_layout()  # Ensure the plot layout is tidy
    plt.show()   

    plotfile="plot_l"+str(lam)+"theobs"+str(theobs)+".png"
    plt.savefig(plotfile)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script_file.py data_file.txt")
    else:
        filename = sys.argv[1]
        main(filename)
