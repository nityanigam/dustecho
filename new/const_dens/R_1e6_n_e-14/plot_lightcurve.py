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
    #"savefig.facecolor": "none",
    "savefig.dpi": 300,

    # Higher DPI for sharper inline images
    "figure.dpi": 96
}

matplotlib.rcdefaults()
matplotlib.rcParams.update(CUSTOM_PLOT_STYLE)

# Define the regular expression pattern
#pattern = r'nH[0-9\.e\+\-]+_lam([0-9\.e\+\-]+)um_theobs([0-9\.e\+\-]+)_Ldnu_xcentr\.txt'
pattern = r"nH(?P<nH>[\d.eE+-]+)_lam(?P<lam>[\d.eE+-]+)um_theobs(?P<theobs>[\d.eE+-]+)"

filenames = ["nH1.0e+00_lam3.37um_theobs0.00_Ldnu_xcentr.txt",
             "nH1.0e+00_lam4.62um_theobs0.00_Ldnu_xcentr.txt"]

colors = ["blue", "green"]

# Function to extract values from filename
def extract_values(filename):
    match = re.match(pattern, filename)
    if match:
        nH = float(match.group("nH"))
        lam = float(match.group("lam"))
        theobs = float(match.group("theobs"))
        return nH, lam, theobs
    else:
        return None, None

# Initialize the plot
fig, ax = plt.subplots()

ax.set_xlabel('Time [days]')
ax.set_ylabel('$L_{\\nu}$ [cgs]')

for i in range(len(filenames)):
    filename = filenames[i]
    if filename.endswith('.txt'):
        nH, lam, theobs = extract_values(filename)
        if nH is not None and lam is not None and theobs is not None:
            print(f'Filename: {filename}, nH: {nH}, lam: {lam}, theobs: {theobs}')
    
    with open(filename, 'r') as file:
        first_line = file.readline()
        first_row_data = first_line.split()[3:]
    
    # Assign the three floats to variables
    tobsmin, tobsmax, Ntobs = map(float, first_row_data)
    
    # Load the data from the text file
    data = np.genfromtxt(filename, dtype=float, skip_header=2, filling_values=np.nan)
    
    Ldnu_cgs = data[0]
    xcentr_pc = data[1]

    log_tobsmin = np.log10(tobsmin)
    log_tobsmax = np.log10(tobsmax)
    
    # Create the logarithmic space with normalized values
    tobs = np.logspace(log_tobsmin, log_tobsmax, num=int(Ntobs), base=10)

    ax.plot(tobs/(60*60*24), Ldnu_cgs, color=colors[i], label="$\\lambda = $"+str(lam))

#plt.title("Spherically distributed dust")
#plt.xlim(left=0, right=1e9/(60*60*24))
plt.legend()
fig.tight_layout()  # Ensure the plot layout is tidy  

plotfile="plot_lam"+str(lam)+".png"
fig.savefig(plotfile)

plt.show()

"""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script_file.py data_file.txt")
    else:
        filename = sys.argv[1]
        main(filename)
"""
