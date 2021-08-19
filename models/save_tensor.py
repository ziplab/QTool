
import torch
import numpy as np
import os
import pdb

def convert_hex(tensor, bitwidth=8, signed=True, ranges=None, check_range=True, debug=False):
    result = []
    length = max(bitwidth - 1, 0) // 4 + 1
    column = tensor.shape[-1]
    tensor_ = tensor.reshape(-1)
    if ranges is None:
        ranges = [-int(pow(2, bitwidth-1)), int(pow(2, bitwidth-1))-1] if signed else [0, int(pow(2, bitwidth))-1]

    for i, data in enumerate(tensor_):
        if check_range and (ranges is not None):
            if data < ranges[0] or data > ranges[1]:
                print('data out of range')
                pdb.set_trace()

        if data < 0:
            data_ = 2**bitwidth + data
            hex_data = '{0:0{1}x}'.format(data_, length)
        else:
            hex_data = '{0:0{1}x}'.format(data, length)
        
        if '0x' in hex_data:
            hex_data = hex_data[2:]
        #print("data: {}, hex_data: {}".format(data, hex_data))

        result.append(hex_data)
        #if (i % column) == (column - 1):
        #    result.append('')

    #pdb.set_trace()
    #print("length of result", len(result))
    return result

def tensor_to_txt(tensor, bitwith, signed=True, filename=None):
    #return True
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    tensor = tensor.astype(np.int32)

    if filename is not None:
        if os.path.isfile(filename):
            return False

        result = convert_hex(tensor, bitwith, signed)
        if len(result) == tensor.size:
            with open(filename, 'a') as f:
                f.write("\n".join(result))
                f.close()
            return True
    return False
        

