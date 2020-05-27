from PIL import Image, ImageDraw
import numpy as np

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

    # normalize 0-> 1
    line = line/255
    return line


def generate_sine(darkness_map, frequency_scale=100, vary_amplitude=False):
    omega_arr = darkness_map * 2 * np.pi * frequency_scale
    amp_scaler = 0.5

    if vary_amplitude:
        amp_scaler = darkness_map * 2 * np.pi * 1
        max_amp = np.max(amp_scaler)
        amp_scaler /= max_amp

    x_arr = np.arange(darkness_map.shape[0])/300
    sine_points = 0.5 * amp_scaler * np.sin(omega_arr * x_arr)
    # max_val = np.max(sine_points)
    # sine_normalized = sine_points / max_val

    return sine_points


def draw_sines(sine_amt, img_matrix):
    # Create an empty image
    out_size = (img_matrix.shape[1], img_matrix.shape[0])
    base = Image.new("L", out_size, color=255)

    # Create canvas to draw on
    canvas = ImageDraw.Draw(base)

    # Calculate the distance between each sine
    delta = (out_size[0]//sine_amt)
    x_axis_steps = np.arange(0, out_size[1])
    # Make an empty points list
    pts_list = []

    # For each sine:
    for current_sine in range(sine_amt):
        # Calculate frequency and amplitude:
        dark_map = collapse_pixels_to_line(img_mat,
                                           current_sine * delta,
                                           (current_sine+1) * delta)

        # Generate the points
        sine_to_draw = generate_sine(dark_map, vary_amplitude=True)
        # Scale the sine to the required size on the y-axis.
        scaled_sine = sine_to_draw*delta
        # Find the center point for the wave to begin, on the y-axis:
        center_y = int(current_sine*delta) + delta/2

        # For "x" pixel, find the corresponding y value from the center:
        for x_val in range(0, len(x_axis_steps)):
            y_val = center_y + scaled_sine[x_val]
            point = (x_val, int(y_val))
            pts_list.append(point)

        # Patchwork, can't be arsed to do the math
        pts_list.append("BREAK")

    # Draw a line between each two consecutive points
    loc = 1
    while loc < len(pts_list)-2:
        if pts_list[loc] == "BREAK":
            # Jump a row if we hit the break point
            loc += 3
        xy = pts_list[loc-1], pts_list[loc]
        canvas.line(xy, fill=25, width=1)
        loc += 1
    return base


if __name__ == "__main__":
    # Configuration
    total_amt_sines = 70

    # End of Configuration
    img_file, img_mat = load_image("mari.jpg")
    output = draw_sines(total_amt_sines, img_mat)
    output.show()
