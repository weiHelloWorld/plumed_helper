import numpy as np

"""a few things to keep in mind:
1. check index: in plumed atom index starts with 1
2. check unit and scaling: in plumed it is nm by default
"""

class Plumed_helper(object):
    def __init__(self): 
        return

    @staticmethod
    def get_atom_positions(index_atoms, scaling_factor, 
                           unit_scaling,     
                           # should explicitly specify unit scaling among different softwares 
                           # (for instance, it is 10.0 if converts from nm in plumed to A in OpenMM pdb file)
                           out_var_prefix='l_0_out_'):
        result = ""
        result += "com_1: COM ATOMS=%s\n" % str(index_atoms)[1:-1].replace(' ', '')
        result += "p_com: POSITION ATOM=com_1\n"

        for item in range(len(index_atoms)):
            result += "p_%d: POSITION ATOM=%d\n" % (item, index_atoms[item])
        # following remove translation using p_com
        for item in range(len(index_atoms)):
            for _1, _2 in enumerate(['.x', '.y', '.z']):
                result += "%s%d: COMBINE PERIODIC=NO COEFFICIENTS=%f,-%f ARG=p_%d%s,p_com%s\n" \
                    % (out_var_prefix, 3 * item + _1, unit_scaling / scaling_factor, unit_scaling / scaling_factor,
                            item, _2, _2)
        return result

    @staticmethod
    def get_pairwise_dis(pair_index, scaling_factor, unit_scaling, 
                           out_var_prefix='l_0_out_'):
        result = ''
        index_input = 0
        for item_1, item_2 in pair_index:
            result += "dis_%d:  DISTANCE ATOMS=%d,%d\n" % (
                index_input, item_1, item_2)
            result += "%s%d: COMBINE PERIODIC=NO COEFFICIENTS=%f ARG=dis_%d\n" % (
                out_var_prefix, index_input, unit_scaling / scaling_factor, index_input)
            index_input += 1
        return result

    @staticmethod
    def shift_scale(in_var_prefix, out_var_prefix, scale_list, offset_list):
        result = ''
        for item, (scale, offset) in enumerate(zip(scale_list, offset_list)):
            result += '%s%d: COMBINE COEFFICIENTS=%f PARAMETERS=%f ARG=%s%d PERIODIC=NO\n' % (
                out_var_prefix, item, scale, -offset / scale, in_var_prefix, item 
            )
        return result
    
    @staticmethod
    def get_minmax_scale(in_var_prefix, out_var_prefix, minmax_scaler):
        # minmax_scaler is a sklearn scaler object
        from sklearn.preprocessing import MinMaxScaler
        assert (isinstance(minmax_scaler, MinMaxScaler))
        return Plumed_helper.shift_scale(in_var_prefix, out_var_prefix, minmax_scaler.scale_, minmax_scaler.min_)

    @staticmethod
    def get_ANN_expression(mode, node_num, ANN_weights, ANN_bias, activation_list):
        result = ''
        if mode == "native":   # using native implementation by PLUMED (using COMBINE and MATHEVAL)
            result += "bias_const: CONSTANT VALUE=1.0\n"  # used for bias
            for layer_index in range(1, len(node_num)):
                for item in range(node_num[layer_index]):
                    result += "l_%d_in_%d: COMBINE PERIODIC=NO COEFFICIENTS=" % (
                        layer_index, item)
                    result += "%s" % \
                                     str(ANN_weights[layer_index - 1][
                                         item * node_num[layer_index - 1]:(item + 1) * node_num[
                                             layer_index - 1]].tolist())[1:-1].replace(' ', '')
                    result += ',%f' % ANN_bias[layer_index - 1][item]
                    result += " ARG="
                    for _1 in range(node_num[layer_index - 1]):
                        result += 'l_%d_out_%d,' % (layer_index - 1, _1)
                    result += 'bias_const\n'

                if activation_list[layer_index - 1].lower() == 'tanh':
                    for item in range(node_num[layer_index]):
                        result += 'l_%d_out_%d: MATHEVAL ARG=l_%d_in_%d FUNC=tanh(x) PERIODIC=NO\n' % (
                            layer_index, item, layer_index, item)
                # generalization for classifier
                elif activation_list[layer_index - 1].lower() == 'softmax':
                    result += "sum_output_layer: MATHEVAL ARG="
                    for item in range(node_num[layer_index]):
                        result += 'l_%d_in_%d,' % (layer_index, item)
                    result = result[:-1] + ' VAR='   # remove last ','
                    for item in range(node_num[layer_index]):
                        result += 't_var_%d,' % item
                    result = result[:-1] + ' FUNC='
                    for item in range(node_num[layer_index]):
                        result += 'exp(t_var_%d)+' % item
                    result = result[:-1] + ' PERIODIC=NO\n'
                    for item in range(node_num[layer_index]):
                        result += 'l_%d_out_%d: MATHEVAL ARG=l_%d_in_%d,sum_output_layer FUNC=exp(x)/y PERIODIC=NO\n' % (
                            layer_index, item, layer_index, item)
                elif activation_list[layer_index - 1].lower() == 'linear':
                    for item in range(node_num[layer_index]):
                        result += 'l_%d_out_%d: MATHEVAL ARG=l_%d_in_%d FUNC=x PERIODIC=NO\n' % (
                            layer_index, item, layer_index, item)
        elif mode == "ANN":  # using ANN class
            temp_num_of_layers_used = len(node_num)
            temp_input_string = ','.join(
                ['l_0_out_%d' % item for item in range(node_num[0])])
            temp_num_nodes_string = ','.join(
                [str(item) for item in node_num[:temp_num_of_layers_used]])
            temp_layer_type_string = ','.join(activation_list)
            temp_coeff_string = ''
            temp_bias_string = ''
            for _1, item_coeff in enumerate(ANN_weights[:temp_num_of_layers_used - 1]):
                temp_coeff_string += ' COEFFICIENTS_OF_CONNECTIONS%d=%s' % \
                                     (_1, ','.join([str(item)
                                                    for item in item_coeff]))
            for _1, item_bias in enumerate(ANN_bias[:temp_num_of_layers_used - 1]):
                temp_bias_string += ' VALUES_OF_BIASED_NODES%d=%s' % \
                    (_1, ','.join([str(item) for item in item_bias]))

            result += "ann_force: ANN ARG=%s NUM_OF_NODES=%s LAYER_TYPES=%s %s %s" % \
                (temp_input_string, temp_num_nodes_string, temp_layer_type_string,
                 temp_coeff_string, temp_bias_string)
        else:
            raise Exception("mode error")
        return result
    
