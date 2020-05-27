from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt


def load_image(path):
    img = Image.open(path)
    img_x = img.size[0]*8
    img_y = img.size[1]*8
    img = img.resize((img_x, img_y))
    img = img.convert("L")  # move to gray scale
    return img, np.asarray(img)


def collapse_pixels_to_line(img_matrix, start_y, end_y):
    w_matrix = np.copy(img_matrix)
    line = 255 - (w_matrix[start_y:end_y].sum(axis=0)//(end_y - start_y))
    line = savgol_filter(line, 7, 3)  # window size 51, polynomial order 3

    # normalize 0-> 1
    line = line/255
    return line


def generate_sine(darkness_map, frequency_scale=100, vary_amplitude=False):
    omega_arr = darkness_map * 2 * np.pi * frequency_scale
    amp_scaler = 1

    if vary_amplitude:
        amp_scaler = darkness_map * 2 * np.pi * 1
        max_amp = np.max(amp_scaler)
        amp_scaler /= max_amp*1.2

    x_arr = np.arange(darkness_map.shape[0])/300
    sine_points = amp_scaler * np.sin(omega_arr * x_arr)
    # max_val = np.max(sine_points)
    # sine_normalized = sine_points / max_val

    return sine_points


def draw_sines(sine_amt, img_mat):

    pts_list = []

    # Create an empty image
    new_image_x = img_mat.shape[1]
    new_image_y = img_mat.shape[0]

    out_size = (new_image_x, new_image_y)

    base = Image.new("L", out_size, color=255)
    canvas = ImageDraw.Draw(base)
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
                pts_list.append(coord)
        pts_list.append("BREAK")



    loc = 1
    while loc < len(pts_list)-2:
        if pts_list[loc] == "BREAK":
            loc += 3
        xy = pts_list[loc-1], pts_list[loc]
        canvas.line(xy, fill=15, width=1)
        loc +=1

    base.show()



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
    total_amt_sines = 100

    # End of Configuration


    img_file, img_mat = load_image("mari.jpg")

    draw_sines(total_amt_sines, img_mat)
    # darkness = np.linspace(0, 255, num=2000, dtype=np.float)
    # collapse_pixels_to_line(img_mat, 0, 5)
    # q = generate_sine(darkness, vary_amplitude=True)

    # plt.plot(darkness, q)
    # plt.show()

    pass