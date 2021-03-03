from PIL import Image, ImageDraw
import numpy as np


def load_image(path, resize=1):
    img = Image.open(path)
    img_x = img.size[0]*resize
    img_y = img.size[1]*resize
    img = img.resize((img_x, img_y))
    img = img.convert("L")  # move to gray scale
    return img, np.asarray(img)


def collapse_pixels_to_line(img_matrix, start_y, end_y):
    w_matrix = np.copy(img_matrix)
    line = 255 - (w_matrix[start_y:end_y].sum(axis=0)//(end_y - start_y))

    # normalize 0-> 1
    line = line/255
    return line


def generate_sine(darkness_map, frequency_scale=100):
    # Get the base frequency based on darkness of our pixels:
    omega_arr = darkness_map * 2 * np.pi * frequency_scale

    # Scale amplitude by pixel darkness as well:
    amp_scaler = darkness_map * 2 * np.pi

    # Normalize 0 -> 1
    max_amp = np.max(amp_scaler)
    amp_scaler /= max_amp

    # Build the time scale:
    x_arr = np.arange(darkness_map.shape[0])/omega_arr

    # Generate final sine as np array:
    sine_points = 0.45 * amp_scaler * np.sin(omega_arr * x_arr)
    return sine_points


def draw_sines(sine_amt, img_matrix, frequency_scale, line_weight):
    # Create an empty image
    out_size = (img_matrix.shape[1], img_matrix.shape[0])
    base_img = Image.new("L", out_size, color=255)

    # Create canvas to draw on
    canvas = ImageDraw.Draw(base_img)

    # Calculate the distance between each sine
    delta = (out_size[1]//sine_amt)
    x_axis_steps = np.arange(0, out_size[0])
    # Make an empty points list
    pts_list = []

    # For each sine:
    for current_sine in range(sine_amt):
        # Calculate frequency and amplitude:
        dark_map = collapse_pixels_to_line(img_mat,
                                           current_sine * delta,
                                           (current_sine+1) * delta)

        # Generate the points
        sine_to_draw = generate_sine(dark_map, frequency_scale=frequency_scale)
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
        canvas.line(xy, fill=25, width=line_weight)
        loc += 1
    return base_img


if __name__ == "__main__":
    # ----------------- Configuration -------------------------------
    total_amt_sines = 50        # Amount of sine waves in the image
    resize_factor = 4           # Scaling factor for the input image
    frequency_factor = 50*np.pi        # Frequency scalar, higher means higher frequency on average
    line_thickness = 1          # Line thickness in pixels
    # ----------------------------------------------------------------

    img_file, img_mat = load_image("mari.png", resize=resize_factor)
    output = draw_sines(total_amt_sines, img_mat, frequency_factor, line_thickness)
    output.show()
