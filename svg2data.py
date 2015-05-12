import numpy as np
import ast
import xml.etree.ElementTree as ET
from operator import itemgetter
import copy
import pylab as plb

class svg2data(object):
    def __init__(self, filename):
        restart = 1
        reason = ''
        while restart:
            restart = 0
            self._tree = ET.parse(filename)
            root = self._tree.getroot()
            width = float(root.attrib['width'])
            height = float(root.attrib['height'])

            # pass transformation to lowest child
            for child in root:
                if reason == 'neg_vals' and 'transform' in child.attrib:
                    del child.attrib['transform']
                child = pass_transformation(child)

            # generate line list and simplify xml
            (root,lines) = get_lines(root)

            for line in lines:
                if (line['min']/width<-0.01).any():
                    if restart == 0:
                        restart = 1
                        reason = 'neg_vals'
                    else:
                        raise Exception('Not able to repair damaged SVG file!')
                    break

        # generate phrases and simplify xml
        (root,phrases) = get_phrases(root)

        # get long lines
        long_lines = get_long_lines(lines,width,height)

        # get axes, graphs and areas
        (axes,graphs,areas) = get_axes_graphs_areas(long_lines)

        # calibrate grid
        self.axes = calibrate_grid(axes,phrases,width,height)

        # calibrate graphs
        graphs = calibrate_graphs(graphs,self.axes)

        # get labels for graphs
        self.graphs = get_labels(graphs,lines,phrases)

    def writesvg(self,filename):
        self._tree.write(filename)

    def plot(self):
        plot_graphs(self.graphs,self.axes)

def transform2matrix(string):
    if 'matrix' in string:
        matrix_string = '['+string[7:-1]+']'
    elif 'scale' in string:
        scale_string = '['+string[6:-1]+']'
        scale = ast.literal_eval(scale_string)
        matrix_string = '('+str(scale[0])+',0,0,'+str(scale[1])+',0,0)'
    elif 'translate' in string:
        trans_string = '['+string[10:-1]+']'
        trans = ast.literal_eval(trans_string)
        matrix_string = '(1,0,0,1,'+str(trans[0])+','+str(trans[1])+')'
    else:
        print(string)
        raise Exception('Not matrix or scale or translate!')

    matrix = np.array(ast.literal_eval(matrix_string)).reshape((3,2)).transpose()
    matrix = np.concatenate((matrix,np.array([[ 0.,  0.,  1.]])), axis=0)
    return matrix

def matrix2transform(matrix):
    return ('matrix('
                    +str(matrix[0][0])+','
                    +str(matrix[1][0])+','
                    +str(matrix[0][1])+','
                    +str(matrix[1][1])+','
                    +str(matrix[0][2])+','
                    +str(matrix[1][2])+')')

def pass_transformation(parent):
    if 'clip-path' in parent.attrib:
        del parent.attrib['clip-path']
    if 'id' in parent.attrib:
        del parent.attrib['id']
    if 'transform' in parent.attrib:
        parent_transform = parent.attrib['transform']
        parent_matrix = transform2matrix(parent_transform)
        for child in parent:
            if 'tspan' in child.tag:
                break
            elif 'transform' in child.attrib:
                old_transform = child.attrib['transform']
                old_matrix = transform2matrix(old_transform)
                new_matrix = np.dot(parent_matrix,old_matrix)
                new_transform = matrix2transform(new_matrix)
                child.attrib['transform'] = new_transform
            else:
                child.attrib['transform'] = parent_transform
            child = pass_transformation(child)
        else:
            if list(parent):
                del parent.attrib['transform']
    else:
        for child in parent:
            if 'tspan' in child.tag:
                break
            child = pass_transformation(child)
    return parent

def parse_path(path_data):
    digit_exp = '0123456789eE'
    comma_wsp = ', \t\n\r\f\v'
    drawto_command = 'MmZzLlHhVvCcSsQqTtAa'
    sign = '+-'
    exponent = 'eE'
    float = False
    entity = ''
    for char in path_data:
        if char in digit_exp:
            entity += char
        elif char in comma_wsp and entity:
            yield entity
            float = False
            entity = ''
        elif char in drawto_command:
            if entity:
                yield entity
                float = False
                entity = ''
            yield char
        elif char == '.':
            if float:
                yield entity
                entity = '.'
            else:
                entity += '.'
                float = True
        elif char in sign:
            if entity and entity[-1] not in exponent:
                yield entity
                float = False
                entity = char
            else:
                entity += char
    if entity:
        yield entity

def path_to_lines_and_curves(path_data): # at the moment curves are ignored
    path = parse_path(path_data)
    drawto_command = 'MmZzLlHhVvCcSsQqTtAa'
    curve_command = 'CcSsQqTtAa'
    command = None
    lines_and_curves = []
    line = []
    coordinate = []
    base_coordinate = [0,0]
    xy = 0
    for entry in path:
        if entry in drawto_command:
            command  = entry
            absolute = entry.isupper()
            if command in 'Mm':
                if absolute:
                    command = 'L'
                else:
                    command = 'l'
                if line:
                    lines_and_curves.append(np.array(line))
                    line = []
            elif command in 'Zz' and line[0] != line[-1]:
                base_coordinate = line[0]
                line.append(line[0])
        elif command not in curve_command:
            if not absolute:
                entry = base_coordinate[xy]+float(entry)
            else:
                entry = float(entry)
            if not xy and command not in 'HhVv':
                coordinate = [entry]
                xy = 1
            else:
                if command in 'Hh':
                    coordinate = [entry, base_coordinate[1]]
                elif command in 'Vv':
                    coordinate = [base_coordinate[0], entry]
                else:
                    coordinate.append(entry)
                base_coordinate = coordinate
                line.append(coordinate)
                xy = 0
    lines_and_curves.append(np.array(line))
    return lines_and_curves

def lines_and_curves_to_path(lines_and_curves): # at the moment curves are ignored
    line_string = ''
    for line in lines_and_curves:
        if line_string:
            line_string += ' '
        line_string += 'M'
        for coordinate in line.flatten():
            line_string += ' '+str(coordinate)
    return line_string

def transform_path(path, matrix):
    points = np.concatenate((path,np.ones((path.shape[0],1))), axis=1)
    new_points = np.einsum('kj,ij',points,matrix).transpose()
    return new_points[:,0:2]

def style2dict(style):
    style = '{\''+style.replace(':','\':\'').replace(';','\',\'')+'\'}'
    return ast.literal_eval(style)

def dict2style(dict):
    style_string = ''
    for k,v in dict.items():
        style_string += k+':'+v+';'
    return style_string[:-1]

def scale_style(style, matrix):
    style_dict = style2dict(style)
    if ('stroke-width' not in style_dict
        and 'stroke-dasharray' not in style_dict
        and 'font-size' not in style_dict):
        return style
    scale_factor = np.sqrt(abs(np.linalg.det(matrix[0:2,0:2])))
    if 'stroke-width' in style_dict:
        style_dict['stroke-width'] = str(
                                float(style_dict['stroke-width'])*scale_factor)
    if ('stroke-dasharray' in style_dict
        and style_dict['stroke-dasharray'] != 'none'
        and style_dict['stroke-dasharray']):
        dasharray = ''
        stroke_dasharray = style_dict['stroke-dasharray'].strip(', ')
        if ' ' not in stroke_dasharray and ',' not in stroke_dasharray:
            stroke_dasharray = stroke_dasharray+','+stroke_dasharray
        for dash in ast.literal_eval(stroke_dasharray):
            dasharray += str(float(dash)*scale_factor)+','
        style_dict['stroke-dasharray'] = dasharray[:-1]
    if 'font-size' in style_dict:
        (size,unit) = font_size2size_units(style_dict['font-size'])
        style_dict['font-size'] = str(size*scale_factor)+unit

    return dict2style(style_dict)

def transform_text(x,y,matrix):
    if matrix[0][1] == 0 and matrix[1][0] == 0:
        x_list = ast.literal_eval('['+x.replace(' ',',')+']')
        y_list = [float(y)]*len(x_list)
        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        x_arr = x_arr*matrix[0][0]+matrix[0][2]
        y_arr = y_arr*matrix[1][1]+matrix[1][2]
        x = " ".join(str(i) for i in x_arr)
        y = " ".join(str(i) for i in y_arr)
        return [x,y]
    else:
        raise Exception('Matrix mixes X and Y axes!')

def font_size2size_units(font_size):
    unit = ''
    for c in reversed(font_size):
        if c not in '0123456789.':
            unit = c+unit
        else:
            break
    size = float(font_size[:-len(unit)])
    return [size, unit]

def chars2phrases(chars):
    chars_sorted = sorted(copy.deepcopy(chars), key=itemgetter('x'), reverse=True)
    phrases = []
    while chars_sorted:
        startchar = chars_sorted.pop()
        phrase = startchar
        del_list = []
        superscript = 0
        subscript = 0
        for i in range(len(chars_sorted)-1, -1, -1):
            testchar = chars_sorted[i]
            if (testchar['x'] < startchar['x']+startchar['size']*0.891
                and testchar['y'] == phrase['y']):
                if superscript or subscript:
                    phrase['text'] += '}'
                    superscript = 0
                    subscript = 0
                phrase['text'] += testchar['text']
                startchar = testchar
                del_list.append(i)
            elif (testchar['x'] < startchar['x']+startchar['size']*.92 # concatenate
                  and testchar['y'] == phrase['y']):
                if superscript or subscript:
                    phrase['text'] += '}'
                    superscript = 0
                    subscript = 0
                phrase['text'] += ' '+testchar['text']
                startchar = testchar
                del_list.append(i)
            elif (testchar['x'] < startchar['x']+startchar['size']
                  and testchar['y'] < phrase['y']
                  and testchar['y'] > phrase['y']-phrase['size']*0.9
                  and testchar['size'] < 0.8*phrase['size']):
                if not superscript:
                    if subscript:
                        phrase['text'] += '}'
                        subscript = 0
                    phrase['text'] += '^{'+testchar['text']
                    superscript = 1
                else:
                    phrase['text'] += testchar['text']
                startchar = testchar
                del_list.append(i)
            elif (testchar['x'] < startchar['x']+startchar['size']
                  and testchar['y']-testchar['size'] < phrase['y']
                  and testchar['y']-testchar['size'] > phrase['y']-phrase['size']*0.9
                  and testchar['size'] < 0.8*phrase['size']):
                if not subscript:
                    if superscript:
                        phrase['text'] += '}'
                        superscript = 0
                    phrase['text'] += '_{'+testchar['text']
                    subscript = 1
                else:
                    phrase['text'] += testchar['text']
                startchar = testchar
                del_list.append(i)
        if superscript or subscript:
            phrase['text'] += '}'
        length = startchar['x']+startchar['size']-phrase['x']
        phrase['dimensions'] = np.array([length,phrase['size']])
        phrase['coords'] = np.array([phrase['x'],phrase['y']])
        del phrase['x']
        del phrase['y']
        del phrase['size']
        phrases.append(phrase)
        for i in del_list:
            del(chars_sorted[i])
    return phrases

def gridlines(axis, line_list, width, height):
    gridlines = []
    if axis['min'][0]==axis['max'][0]:
        axis_type = 0
    elif axis['min'][1]==axis['max'][1]:
        axis_type = 1
    else: return False
    for line in line_list:
        length = line['max'][axis_type]-line['min'][axis_type]
        if (line['max'][axis_type]>=axis['min'][axis_type]
            and line['min'][axis_type]<=axis['min'][axis_type]+length/10
            and line['min'][1-axis_type]==line['max'][1-axis_type]
            and length < width/20
            and length < height/20
           ):
            gridline = {}
            gridline['length'] = length
            gridline['d'] = line['min']
            gridlines.append(gridline)
    return gridlines

def get_lines(root):
    lines = []
    paths = []
    for path in reversed(list(root.iter('{http://www.w3.org/2000/svg}path'))):
        if 'transform' in path.attrib:
            matrix = transform2matrix(path.attrib['transform'])
            line_list = []
            # transform path coordiantes
            for line in path_to_lines_and_curves(path.attrib['d']):
                line_list.append(transform_path(line, matrix))
            # transform path style
            style = scale_style(path.attrib['style'], matrix)
            del path.attrib['transform']
        else:
            line_list = path_to_lines_and_curves(path.attrib['d'])
            if 'style' in path.attrib:
                style = path.attrib['style']
        path.attrib['style'] = style
        path.attrib['d'] = lines_and_curves_to_path(line_list)
        if path.attrib not in paths:
            paths.append(path.attrib)
            # extract only information needed for data extraction
            for line in line_list:
                max_values = np.amax(line, axis=0)
                min_values = np.amin(line, axis=0)
                line_dict = {'d':line, 'style':style2dict(style), 'max':max_values, 'min':min_values}
                lines.append(line_dict)
        else:
            path.clear()
    return(root,lines)

def get_phrases(root):
    chars = []
    for text in root.iter('{http://www.w3.org/2000/svg}text'):
        if 'transform' in text.attrib:
            matrix = transform2matrix(text.attrib['transform'])
            if not (matrix[0][1] == 0 and matrix[1][0] == 0):
                scalerotmatrix = matrix[0:2,0:2]
                translatematrix = matrix[0:2,2:3]
                scale_factor = np.sqrt(abs(np.linalg.det(scalerotmatrix)))
                rotationmatrix = scalerotmatrix/scale_factor
                rotationmatrix = np.concatenate((rotationmatrix,
                                                 np.zeros((2, 1))),axis=1)
                rotationmatrix = np.concatenate((rotationmatrix,
                                                 np.array([[ 0.,  0.,  1.]])),
                                                axis=0)
                translatematrix = np.dot(rotationmatrix[0:2,0:2].transpose(),
                                                            translatematrix)
                matrix = np.identity(2)*scale_factor
                matrix = np.concatenate((matrix,translatematrix), axis=1)
                matrix = np.concatenate((matrix,np.array([[ 0.,  0.,  1.]])),
                                                                     axis=0)
                text.attrib['transform'] = matrix2transform(rotationmatrix)
            else:
                del text.attrib['transform']
            for tspan in text:
                (x,y)=transform_text(tspan.attrib['x'],tspan.attrib['y'],matrix)
                x_list = ast.literal_eval('['+x.replace(' ',',')+']')
                y_list = ast.literal_eval('['+y.replace(' ',',')+']')
                style = scale_style(tspan.attrib['style'], matrix)
                style_dict = style2dict(style)
                text_attr = tspan.text
                if text_attr == None:
                    text_attr = ''
                (size,unit) = font_size2size_units(style_dict['font-size'])
                for char_zip in zip(text_attr, x_list, y_list):
                    char = {'text':char_zip[0],'x':char_zip[1],'y':char_zip[2],
                                                                    'size':size}
                    chars.append(char)
                tspan.attrib['x'] = x
                tspan.attrib['y'] = y
                tspan.attrib['style'] = style
    phrases = chars2phrases(chars)
    return [root,phrases]

def get_long_lines(lines,width,height):
    lhls = []
    lvls = []
    for line in lines:
        if line['max'][0]-line['min'][0] > width/2:
            grid = gridlines(line, lines, width, height)
            if grid:
                line['grid']=grid
            lhls.append(line)
        elif line['max'][1]-line['min'][1] > height/2:
            grid = gridlines(line, lines, width, height)
            if grid:
                line['grid']=grid
            lvls.append(line)
    return [lhls,lvls]

def get_axes_graphs_areas(long_lines):
    axes = [[],[]]
    areas = []
    mesh = []
    graphs_prel = []
    graphs = []
    for axis_type in range(2):
        for long_line in long_lines[axis_type]:
            if 'grid' in long_line:
                axes[axis_type].append(long_line)
            elif ((long_line['style']['fill'] and long_line['style']['fill'] != 'none')
                  or (long_line['d'][0] == long_line['d'][-1]).all()):
                areas.append(long_line)
            elif (long_line['max'][1-axis_type] == long_line['min'][1-axis_type]
                 and long_line['style']['stroke-dasharray']
                 and long_line['style']['stroke-dasharray'] != 'none'):
                mesh.append(long_line)
            elif axis_type == 0:
                graphs_prel.append(long_line)
    for graph_prel in graphs_prel:
        boundary = 0
        if graph_prel['max'][1] == graph_prel['min'][1]:
            for axis in axes[1]:
                if (axis['min'][1]+float(graph_prel['style']['stroke-width'])+2
                    > graph_prel['max'][1]):
                    boundary = 1
        if boundary == 0:
            graphs.append(graph_prel)
    return [axes,graphs,areas]

def calibrate_grid(axes,phrases,width,height):
    grids_calibr = [{},{}]
    for axis_type in range(2):
        for axis in axes[axis_type]:
            if ((axis['min'][1] > height/2 and axis_type == 0)
                or
                (axis['min'][0] < width/2 and axis_type == 1)):
                if grids_calibr[axis_type]:
                    break
                grid_calibr = []
                grid_sorted = sorted(axis['grid'], key=itemgetter('length'), reverse=True)
                for phrase in phrases:
                    if ((phrase['coords'][1] > axis['min'][1]+phrase['dimensions'][1]*1.1 # x-axis label
                         and axis_type == 0)
                        or
                        (phrase['coords'][0]+phrase['dimensions'][0]*1.2 < axis['min'][0] # y-axis label
                         and phrase['coords'][1] < axis['min'][0]
                         and axis_type == 1)):
                      if 'name' in grids_calibr[axis_type]:
                          grids_calibr[axis_type]['name'] += phrase['text']
                      else:
                          grids_calibr[axis_type]['name'] = phrase['text']
                for gridline in grid_sorted:
                    for phrase in phrases:
                        if (((phrase['coords'][1] > gridline['d'][1]+gridline['length']
                              and phrase['coords'][0] < gridline['d'][0]
                              and phrase['coords'][0]+phrase['dimensions'][0] > gridline['d'][0]
                              and phrase['coords'][1] < axis['min'][1]+phrase['dimensions'][1]*1.1
                             ) and axis_type == 0)
                            or
                            ((phrase['coords'][0]+phrase['dimensions'][0]*0.7 < gridline['d'][0]
                              and phrase['coords'][1]-phrase['dimensions'][1] < gridline['d'][1]
                              and phrase['coords'][1] > gridline['d'][1]
                              and phrase['coords'][0] > gridline['d'][0]-phrase['dimensions'][0]*1.5
                             ) and axis_type == 1)
                            ):
                            if (grid_calibr and gridline['length'] < grid_calibr[-1]['length']
                                and len(grid_calibr) >=3):
                                break
                            else:
                                gridline['text'] = phrase['text']
                                text = gridline['text'];
                                if '^' in text:
                                    text = text.replace('}','')
                                    text = text.replace('0^{','E')
                                gridline['value'] = float(text)
                                for gridtest in grid_calibr:
                                    if (gridtest['d'] == gridline['d']).all():
                                        break
                                else:
                                    grid_calibr.append(gridline)
                if grid_calibr:
                    value0 = grid_calibr[0]['value']
                    value1 = grid_calibr[1]['value']
                    value2 = grid_calibr[2]['value']
                    distance10 = (grid_calibr[1]['d'][axis_type]
                                 -grid_calibr[0]['d'][axis_type])
                    distance21 = (grid_calibr[2]['d'][axis_type]
                                 -grid_calibr[1]['d'][axis_type])
                    if abs((value1-value0)/(value2-value1)
                           /distance10*distance21-1)<0.1:
                        axis_scaling = 'linear'
                    elif abs((np.log(value1)-np.log(value0))
                             /(np.log(value2)-np.log(value1))
                             /distance10*distance21-1)<0.1:
                        axis_scaling = 'log'
                        for gridline in grid_calibr:
                            gridline['value'] = np.log(gridline['value'])
                    else:
                        print(grid_calibr)
                        raise Exception('Did not find axis scaling!')

                    grids_calibr[axis_type]['type']=axis_scaling
                    grids_calibr[axis_type]['grid']=grid_calibr
    return grids_calibr

def get_calibr_matrix(grids_calibr):
    matrix = np.zeros((3, 3))
    matrix[2][2] = 1
    for axis_type in range(2):
        svg = []
        axis = []
        grid_calibr = grids_calibr[axis_type]
        for grid in grid_calibr['grid']:
            svg.append(grid['d'][axis_type])
            axis.append(grid['value'])
        svg = np.array(svg)
        axis = np.array(axis)
        d_svg = svg[0]-svg[-1]
        d_axis = axis[0]-axis[-1]
        scale_factor = d_axis/d_svg
        translation = (axis[0]-svg[0]*scale_factor)
        matrix[axis_type][axis_type] = scale_factor
        matrix[axis_type][2] = translation
    return matrix

def calibrate_path(path,grids_calibr):
    matrix = get_calibr_matrix(grids_calibr)
    new_path_tr = transform_path(path, matrix).transpose()
    for axis_type in range(2):
        if grids_calibr[axis_type]['type'] == 'log':
            new_path_tr[axis_type] = np.exp(new_path_tr[axis_type])
    return new_path_tr

def calibrate_graphs(graphs,grids_calibr):
    for graph in graphs:
        path = graph['d']
        path_calib = calibrate_path(path,grids_calibr)
        graph['values'] = path_calib
    return graphs

def get_labels(graphs,lines,phrases):
    new_graphs = []
    for graph in graphs:
        graph['label'] = ''
        graph['label_len'] = 0
        for line in lines:
            if ((line['style'] == graph['style'])
                and line['min'][1] == line['max'][1]
                and not np.array_equal(line['d'], graph['d'])):
                for phrase in phrases:
                    if (phrase['coords'][0] > line['max'][0]
                        and phrase['coords'][0]-phrase['dimensions'][1]*2 < line['max'][0]+graph['label_len']*0.92
                        and phrase['coords'][1]>line['min'][1]
                        and phrase['coords'][1]-phrase['dimensions'][1]*0.7<line['min'][1]):
                        graph['label'] += ' '+phrase['text']
                        graph['label_len'] += phrase['dimensions'][0]
        graph['label']=graph['label'].strip()
    for graph in graphs:
        doubleline = 0
        if graph['label']=='':
            for cmp_graph in graphs:
                if (np.array_equal(cmp_graph['d'], graph['d'])
                    and cmp_graph['style'] != graph['style']
                    and cmp_graph['label'] != ''):
                    doubleline = 1
            for cmp_graph in new_graphs:
                if np.array_equal(cmp_graph['d'], graph['d']):
                    doubleline = 1
        if not doubleline:
            new_graphs.append(graph)
    graphs = new_graphs
    return graphs

def plot_graphs(graphs,grids_calibr):
    plt = plb.plt
    for graph in graphs:
        if 'label' in graph and graph['label'] != '':
            labeltext = graph['label']
            labeltext = labeltext.replace(' ','\ ').replace('%','\%').replace('±','\pm')
        else:
            labeltext = 'no\ label'
        line = plt.plot(graph['values'][0],
                        graph['values'][1],
                        graph['style']['stroke'],
                        label=r'$'+labeltext+'$')
        if 'stroke-dasharray' in graph['style'] and graph['style']['stroke-dasharray'] != 'none':
            plb.plt.setp(line, dashes=[2,2])
    #plt.title('title')
    xlabel = grids_calibr[0]['name'].replace(' ','\ ').replace('%','\%')
    ylabel = grids_calibr[1]['name'].replace(' ','\ ').replace('%','\%')
    plt.xlabel(r'$'+xlabel+'$')
    plt.ylabel(r'$'+ylabel+'$')
    plt.xscale(grids_calibr[0]['type'])
    plt.yscale(grids_calibr[1]['type'])
    plt.rcParams['legend.loc'] = 'best'
    plt.legend();
    plt.show()
