import os
import matplotlib.pyplot as plt
from gdal import ogr

# Step 1: Read OCAD files from directory
input_directory = "path/to/ocad/files"  # Replace with your input directory
ocad_files = [file for file in os.listdir(input_directory) if file.endswith(".ocd")]

# Step 2: Create an empty list for coordinates
coordinates_list = []

# Step 3-6: Process each OCAD file
for ocad_file in ocad_files:
    file_path = os.path.join(input_directory, ocad_file)

    # Step 1-2: Open file and find coordinate information
    ocad_ds = ogr.Open(file_path)
    layer = ocad_ds.GetLayer()

    # Step 3: Calculate rectangular vertices
    envelope = layer.GetExtent()
    x_min, x_max, y_min, y_max = envelope[0], envelope[1], envelope[2], envelope[3]

    # Step 4: Add rectangular coordinates to the list
    coordinates_list.append([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

# Step 7: Plot rectangular areas on a geographical map
fig, ax = plt.subplots()
for coordinates in coordinates_list:
    # Extract x and y coordinates separately
    x_coords, y_coords = zip(*coordinates)

    # Plot rectangular area
    ax.plot(x_coords + (x_coords[0],), y_coords + (y_coords[0],), marker="o")

# Customize the plot as desired
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Rectangular Areas on a Geographical Map")
ax.grid(True)

# Display the plot
plt.show()
