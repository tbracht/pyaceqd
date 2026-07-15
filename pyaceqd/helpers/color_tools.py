import colorsys

def hex_to_rgba(hex_code):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 6:
        hex_code += 'FF'  # Append alpha channel if not provided
    decimal_value = int(hex_code, 16)
    rgba_tuple = (decimal_value >> 24 & 255, decimal_value >> 16 & 255, decimal_value >> 8 & 255, decimal_value & 255)
    return rgba_tuple

def select_equally_spaced_colors(n):
    colors = []
    hue_values = [i / n for i in range(n)]  # Equally spaced hue values
    # using HLS color space: L is lightness, S is saturation. L=0.5 and S=1.0 for bright colors
    # Use a slightly muted HLS palette so the colors stay readable in plots.
    for hue in hue_values:
        rgb = colorsys.hls_to_rgb(h=hue, l=0.58, s=0.78)  # Convert HSL to RGB. Also see https://en.wikipedia.org/wiki/HSL_and_HSV
        hex_code = "#{:02X}{:02X}{:02X}".format(*[int(255 * c) for c in rgb])  # Convert RGB to hexadecimal color code
        colors.append(hex_code)
    return colors