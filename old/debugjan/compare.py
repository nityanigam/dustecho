import numpy as np

def read_3d_array_from_file(filename):
    """Reads the 3D array from a file in the format provided."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize parameters
    Nt = Nr = Na = None
    reading_data = False
    array = []
    temp_layer = []

    for line in lines:
        line = line.strip()

        # Find Nt, Nr, Na based on header information
        if line.startswith('tmin'):
            _, _, _, tmin, tmax, Nt, _ = line.split()
            Nt = int(Nt)
        elif line.startswith('rmin'):
            _, _, _, rmin, rmax, Nr, _ = line.split()
            Nr = int(Nr)
        elif line.startswith('amin'):
            _, _, _, amin, amax, Na, _ = line.split()
            Na = int(Na)
        elif line.startswith('i='):
            # New time slice (Nt axis)
            if reading_data and temp_layer:
                array.append(temp_layer)
            temp_layer = []
            reading_data = True
        elif reading_data:
            # Read the row of data (Na elements per row)
            row_data = list(map(float, line.split()))
            temp_layer.append(row_data)
    
    # Add the last layer
    if temp_layer:
        array.append(temp_layer)
    
    return np.array(array, dtype=np.float64)

def compare_arrays(array1, array2):
    """Compares two 3D arrays and returns the absolute difference."""
    if array1.shape != array2.shape:
        raise ValueError("Arrays have different shapes and cannot be compared.")

    difference = np.abs(array1 - array2)
    return difference

def main():
    # File paths (update to actual paths if necessary)
    file_Td = 'nH1.0e+00_Td.txt'
    file_Tavg = 'nH1.0e+00_Tavg.txt'

    # Read the arrays
    Td_array = read_3d_array_from_file(file_Td)
    Tavg_array = read_3d_array_from_file(file_Tavg)

    # Compare the arrays
    difference_array = compare_arrays(Td_array, Tavg_array)

    # Output the comparison result
    print("Maximum difference between arrays: ", np.max(difference_array))
    print("Average difference between arrays: ", np.mean(difference_array))

    # Optionally, save the difference array to a file or visualize
    np.savetxt('array_difference.txt', difference_array.reshape(-1), fmt='%.8e')
    print("Difference array saved to 'array_difference.txt'.")

if __name__ == "__main__":
    main()
