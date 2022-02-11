import numpy as np

def encode_segmap(mask, mapping, ignore_index):
    label_copy = ignore_index * np.ones(mask.shape, dtype=np.float32)
    for k, v in mapping:
        label_copy[mask == k] = v

    return label_copy

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = image.astype(int)

    final = np.zeros((x.shape[0], x.shape[1],3), np.ubyte)
    x[x==255]=19
    final[:,:]=colour_codes[x]
    return final