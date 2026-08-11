import math
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
from IPython.display import display, HTML

ROOT = Path(__file__).resolve().parent.parent.parent 

def show_asset_images(df, asset_id, max_cols=4, figsize=(6, 6)):
    """
    Display all images associated with a given asset_id in a matplotlib grid.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that must contain at least the columns:
        - 'asset_id': identifier for grouping images
        - 'image_path': relative image file paths

    asset_id : int or str
        The ID of the asset whose images will be displayed.

    max_cols : int, default=4
        Maximum number of image columns in the grid layout.

    figsize : tuple(int, int), default=(6, 6)
        Figure size passed to `matplotlib.pyplot.subplots`.

    Returns
    -------
    None
        Displays a matplotlib figure of the image grid.
    """

    # Get paths
    paths = (
        df.loc[df['asset_id'] == asset_id, "image_path"]
        .dropna()
        .tolist()
    )

    if len(paths) == 0:
        print("No images found for this asset.")
        return

    fixed_paths = [str(ROOT / p.replace("data/", "data/raw/", 1)) for p in paths]

    # Grid layout
    n = len(fixed_paths)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easier indexing
    axes = axes.flatten() if n > 1 else [axes]

    for ax, img_path in zip(axes, fixed_paths):
        if not os.path.exists(img_path):
            ax.set_title("Missing")
            ax.axis("off")
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # correct orientation

        ax.imshow(img)
        ax.axis("off")

    # Turn off any extra axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
def create_image_tiles(df, image_col="image_path", thumb_size=200):
    """
    Create an inline HTML tile grid showing small thumbnails for each row in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata. Must include:
        - image_col: string column with image paths
        - optional: 'asset_id' for labeling each tile

    image_col : str, default="image_path"
        Name of the column in the DataFrame that contains raw image paths.

    thumb_size : int, default=200
        Pixel size for thumbnail width and height.

    Notes
    -----
    - Paths are modified by replacing 'data/citywide' with '../data/raw/citywide'.

    Returns
    -------
    None
        Displays an HTML grid of image thumbnails.
    """

    tiles_html = ""

    for _, row in df.iterrows():
        path = row[image_col].replace("data/citywide", "../data/raw/citywide")

        label = str(row.get("asset_id", os.path.basename(path)))

        tiles_html += f"""
        <div style="display:flex; flex-direction:column; align-items:center; margin:6px;">
            <img src="{path}"
                 style="width:{thumb_size}px; height:{thumb_size}px; object-fit:cover;
                        border-radius:6px; border:1px solid #ddd;"/>
            <span style="font-size:11px;">{label}</span>
        </div>
        """

    html = f"""
    <div style="display:flex; flex-wrap:wrap; gap:8px; padding:12px;">
        {tiles_html}
    </div>
    """

    display(HTML(html))