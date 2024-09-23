import random
import os

from PIL import Image
import numpy as np
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, column
from bokeh.models import CustomJS, Div, TapTool
output_notebook()

def image_and_descriptions_plot(image_files, n_columns, n_rows):

    def preprocess_image(image, target_width=300):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        width, height = image.size

        f_scale = target_width / width
        new_size = (int(width * f_scale), int(height * f_scale))
        image = image.resize(new_size)

        return image.convert("RGBA")

    # define text block (div) that will show the image description when clicked
    text_box = Div(text="Image description", 
                width=400, height=50, visible=True, name='text-box',
                styles={'background-color': '#FFFFA5',  # Light yellow
                        'color': 'black', 
                        'padding': '10px', 
                        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                        'font-size': '16px',
                        'border-radius': '5px',
                        'display': 'flex',
                        'align-items': 'center',
                        'justify-content': 'center',
                        'position': 'absolute',
                        'top': '100px',
                        'left': '50%',
                        'transform': 'translateX(-50%)',
                        'z-index': '1000',
                        'display': 'block',
                        'height': 'auto',
                        'word-wrap': 'break-word'})


    # only read the necessary images
    image_data = [preprocess_image(Image.open(imf)) for imf in image_files[:n_columns*n_rows]]
    image_descriptions = []
    for imf in image_files[:n_columns*n_rows]:
        with open(imf + ".txt") as f:
            image_descriptions.append(f.read())


    figures = []
    for i, image in enumerate(image_data):
        p = figure(width=image.size[0], height=image.size[1], tools="tap")
        p.background_fill_color = "#1A1A1A"
        p.border_fill_color = "#1A1A1A"
        p.outline_line_color = None

        p.min_border_left = 0
        p.min_border_right = 0
        p.min_border_top = 0
        p.min_border_bottom = 0
        data = np.array(image)
        p.image_rgba(image=[data.view("uint32").reshape(data.shape[:2])], x=0, y=0, dw=data.shape[0], dh=data.shape[1])

        p.axis.visible = False
        p.grid.visible = False

        tap_tool = p.select(type=TapTool)[0]
        tap_tool.callback = CustomJS(args=dict(text_box=text_box, 
                                            image_description = image_descriptions[i]), code="""
            text_box.text = image_description;
        """)

        figures.append(p)

    grid_figures = [figures[i:i+n_columns] for i in range(0, len(figures), n_columns)]
    grid = gridplot(grid_figures)

    layout = column(grid, text_box, sizing_mode='fixed')
    return layout