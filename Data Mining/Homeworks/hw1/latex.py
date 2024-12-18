import os

folder_path = '/Users/davide/Desktop/HW - 1 - DM/imgs/'  # Folder containing images
images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
images.sort()

# Calculate minipage width based on the number of images
minipage_width = 1 / len(images)  # Ensure images fit in one row

latex_code = r"""
\begin{figure}[h!]
    \centering
"""

for img in images:
    latex_code += f"""
    \\begin{{minipage}}{{{minipage_width:.2f}\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{imgs/{img}}}
        \\caption{{Image: {img}}}
    \\end{{minipage}}
    """

latex_code += r"""
\end{figure}
"""

print(latex_code)
