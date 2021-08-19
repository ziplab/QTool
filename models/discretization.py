
import numpy as np
import torch
import pdb

from . import save_tensor as st

def discretize(x, alpha, step, level, bit_limit, global_buffer=None, index=0, input_index='', closed_form=False, debug_=False):
    shifts = []
    CF = []
    offset = []
    scale_limit = 10e-19
    level = int(level)
    y = torch.zeros_like(x)
    for i, scale in enumerate(alpha):
        scale = abs(scale)
        if scale > scale_limit:
            eta_bn = x[0][i].div(scale) if len(alpha) != 1 else x.div(scale)
            eta_bn = torch.round(eta_bn) # 19bit unsigned

            for shift in range(20):
                max_range = int(step / scale * (level - 0.)) + 0.
                if max_range >= pow(2., bit_limit):
                    scale = scale * 2.
                else:
                    break
            shifts.append(shift)

            if int(max_range) > pow(2., bit_limit):
                eta_bn.fill_(0.0)
                offset.append(0)
                CF.append(255)
                shifts[-1] = 20
                continue
                pdb.set_trace()

            def get_cthk(CTHK, uthk, dthk, tbit, cbit, last_index=0, keep_precision=False, verbose=False):
                if cbit >= tbit:
                    return
                    
                index = last_index + int(pow(2., tbit - 1 - cbit))
                cthk = (uthk + dthk) / 2. if keep_precision else (uthk + dthk) // 2.
                CTHK[index] = int(cthk)
                if verbose:
                    print("index: {}, level: {} -> {:0.1f} vs {:0.1f}".format(
                        index, cbit, ((index - 0.5) * step / scale), cthk))
                get_cthk(CTHK, cthk, dthk, tbit, cbit + 1, last_index, keep_precision, verbose)
                get_cthk(CTHK, uthk, cthk, tbit, cbit + 1, index, keep_precision, verbose)

            def test_cf(CTHK, up=1, down=0, keep_precision=False, verbose=False):
                get_cthk(CTHK, max_range + up, down, round(np.log2(level)), 0, 0, keep_precision=keep_precision, verbose=verbose)

            def get_error(CTHK, verbose=False):
                e1, e2, e3, e4, e5 = 0, 0, 0, 0, 0
                for t in range(1, int(level)):
                    e1 = e1 + abs(CTHK[t] - int((t - 0.5) * step / scale) - round(0.5*step/scale) - 2)
                    e2 = e2 + abs(CTHK[t] - int((t - 0.5) * step / scale) - round(0.5*step/scale) - 1)
                    e3 = e3 + abs(CTHK[t] - int((t - 0.5) * step / scale) - round(0.5*step/scale) - 0)
                    e4 = e4 + abs(CTHK[t] - int((t - 0.5) * step / scale) - round(0.5*step/scale) + 1)
                    e5 = e5 + abs(CTHK[t] - int((t - 0.5) * step / scale) - round(0.5*step/scale) + 2)
                if verbose:
                    print(e1, e2, e3, e4, e5)
                return min(e1, e2, e3, e4, e5), [e1, e2, e3, e4, e5]

            if not closed_form:
                if 'cthk-{}-{}-{}'.format(index, input_index, i) not in global_buffer:
                    CTHK = np.zeros(int(level))
                    last_error = 15
                    kp = False
                    for case in [(0, 0), (1, 0), (2, 0)]: #, (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]:
                        up, down = case
                        test_cf(CTHK, up, down, keep_precision=kp)
                        error, _ = get_error(CTHK)
                        if error < last_error:
                            last_case, last_error = case, error
                    #print("{} > {}".format(last_case, last_error))
                    test_cf(CTHK, last_case[0], last_case[1], keep_precision=kp)
                    _, errors = get_error(CTHK)
                    cthk_offset = errors.index(last_error) - 2
                    max_range = max_range + last_case[0]
                    assert int(max_range) < pow(2., bit_limit)
                    cthk_offset = cthk_offset - round(0.5*step/scale)
                    global_buffer['cthk-{}-{}-{}'.format(index, input_index, i)] = CTHK
                    global_buffer['cthk_offset-{}-{}-{}'.format(index, input_index, i)] = cthk_offset
                else:
                    CTHK = global_buffer['cthk-{}-{}-{}'.format(index, input_index, i)]
                    cthk_offset = global_buffer['cthk_offset-{}-{}-{}'.format(index, input_index, i)]

            if False:
                def my_ceil(a, precision=0):
                    return np.true_divide(np.ceil(a * 10**precision), 10**precision)
                
                def my_floor(a, precision=0):
                    return np.true_divide(np.floor(a * 10**precision), 10**precision)

                CTHK = np.zeros(int(level))
                a = step/scale
                for bit in range(1, 5):
                    error1 = [0.5*a  - pow(2., my_floor(np.log2(a*0.5), bit)), 0.5*a - pow(2., my_ceil(np.log2(a*0.5), bit))]
                    error2 = [pow(2., my_floor(np.log2(a*15.5), bit)) - a*15.5, pow(2., my_ceil(np.log2(a*15.5), bit)) - a*15.5]
                    error = min(abs(error1[0] + error2[0]), abs(error1[0] + error2[1]), abs(error1[1] + error2[0]), abs(error1[1] + error2[1]))
                    print(error, error1, error2, bit, my_floor(np.log2(a*15.5), bit), my_floor(np.log2(a*0.5), bit))
                    if error < 1/14.:
                        break
                print("bit: {}, a: {}, 0.5a:{}, 15.5a: {}".format(bit, a, 0.5*a, 15.5*a))
                dthk = - pow(2., np.round(np.log2(a*0.5), bit))
                uthk = pow(2., np.round(np.log2(a*15.5), bit))
                get_cthk(CTHK, uthk, dthk, 4, 0, last_index=0, keep_precision=True, verbose=True)
                pdb.set_trace()

            if not closed_form:
                bias = cthk_offset *  pow(2, shift)
                if bias >= pow(2., 15) or bias < -pow(2., 15):
                    eta_bn.fill_(0.0)
                    offset.append(0)
                    CF.append(255)
                    shifts[-1] = 20
                    continue
                offset.append(cthk_offset)
                eta_bn = eta_bn - bias
                cthk_offset = 0
            else:
                offset.append(0)
            CF.append(int(max_range))

            #if 'print-{}'.format(input_index) in global_buffer:
            #    st.tensor_to_txt(eta_bn, 19, signed=False, \
            #        filename='cmodel/discretization-{}_{}-before-quant_uint19.hex'.format(index, input_index))

            if debug_:
                print("i-a", i)
                pdb.set_trace()

            eta_bn = eta_bn.div(pow(2., shift))
            eta_bn = torch.round(eta_bn)
            eta_bn = torch.clamp(eta_bn, min=0, max=pow(2., bit_limit) -1)

            if debug_:
                print("i-b", i)
                pdb.set_trace()

            for threshold in range(1, int(level)):
                if closed_form:
                    level_clip = int((threshold - 0.5) * step/scale)
                else:
                    level_clip = CTHK[threshold] + cthk_offset

                if len(alpha) != 1:
                    y[0][i].masked_fill_(eta_bn > level_clip, threshold)
                else:
                    y.masked_fill_(eta_bn > level_clip, threshold)

            #if 'print-{}'.format(input_index) in global_buffer:
            #    save = y[0][i] if len(alpha) != 1 else y
            #    st.tensor_to_txt(save, 4, signed=False, \
            #        filename='cmodel/discretization-{}_{}-after-quant_uint4.hex'.format(index, input_index))
            if debug_:
                print("i-c", i)
                pdb.set_trace()
            #pdb.set_trace()
        else:
            shifts.append(20)
            CF.append(255)
            offset.append(0)
    return y, shifts, CF, offset

