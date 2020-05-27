from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt


def load_image(path):
    img = Image.open(path)
    img_x = img.size[0]*4
    img_y = img.size[0]*4
    img = img.resize((img_x, img_y))
    img = img.convert("L")  # move to gray scale
    return img, np.asarray(img)


def collapse_pixels_to_line(img_matrix, start_y, end_y):
    w_matrix = np.copy(img_matrix)
    line = 255 - (w_matrix[start_y:end_y].sum(axis=0)//(end_y - start_y))
    line[line < 5] = 3
    return line


def generate_sine(darkness_map, frequency_scale=10, vary_amplitude=False):
    omega_arr = darkness_map * 2 * np.pi * frequency_scale
    amp_scaler = 1

    if vary_amplitude:
        amp_scaler = darkness_map * 2 * np.pi * 1

    x_arr = np.arange(darkness_map.shape[0])/20000
    sine_points = amp_scaler*np.sin(omega_arr * x_arr)
    max_val = np.max(sine_points)
    sine_normalized = sine_points / max_val

    return sine_normalized


def draw_sines(sine_amt, img_mat):
    # Create an empty image
    new_image_x = img_mat.shape[1]
    new_image_y = img_mat.shape[0]

    out_size = (new_image_x, new_image_y)

    canvas = Image.new("L", out_size, color=255)
    delta = (out_size[0]//sine_amt)

    sample_rate = np.arange(0, out_size[1])

    for current_sine in range(sine_amt-1):
        dark_map = collapse_pixels_to_line(img_mat,
                                           current_sine * delta,
                                           (current_sine+1) * delta)
        sine_to_draw = generate_sine(dark_map, vary_amplitude=True)
        scaled_sine = sine_to_draw*delta


        # DRAW:
        center_y = int(current_sine*delta) + delta/2
        for pt in range(0, len(sample_rate)):
            y_val = center_y + scaled_sine[pt]
            if np.isnan(y_val):
                print("nan")
            else:
                coord = (pt, int(y_val))
                canvas.putpixel(coord, 0)
    canvas.show()



if __name__ == "__main__":
    #
    # Load image
    # Get parameters from configuration:
    #   - how many waves in the image
    #       - this equals the interval (in pixels) where we can center the starts of waves
    #   - ratio of frequency/darkness
    #   - change amplitude?
    # Draw sines with respect to pixel size, changing in frequency

    # Configuration
    total_amt_sines = 50

    # End of Configuration


    img_file, img_mat = load_image("calibration_image.png")

    draw_sines(total_amt_sines, img_mat)
    # darkness = np.linspace(0, 255, num=2000, dtype=np.float)
    # collapse_pixels_to_line(img_mat, 0, 5)
    # q = generate_sine(darkness, vary_amplitude=True)

    # plt.plot(darkness, q)
    # plt.show()

    pass