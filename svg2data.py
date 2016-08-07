import numpy as np
import ast
import xml.etree.ElementTree as ET
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.afm import AFM
import os.path
afm_dir = os.path.join(rcParams['datapath'],'fonts', 'afm')
afm_dict = {}
for afm_file in os.listdir(afm_dir):
    afm_fname = os.path.join(afm_dir,afm_file)
    afm = AFM(open(afm_fname, 'rb'))
    afm_dict[afm.get_fullname().lower()] = afm_fname
afm_translate = {'arial':'helvetica',
                 'arial bold':'helvetica bold',
                 'times':'times roman',
                 'times new roman':'times roman',
                 'times new roman bold':'times bold',}

class svg2data(object):
    def __init__(self, filename, test=False, debug=None):
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
            if reason == 'neg_vals':
                self._tree.write(filename)

            # generate line list and simplify xml
            (root,lines,curves) = get_lines_curves(root)

            for line in lines:
                if (line['min']/width<-0.01).any():
                    if reason == '':
                        restart = 1
                        reason = 'neg_vals'
                    else:
                        raise Exception('Not able to repair damaged SVG file!')
                    break

        # generate phrases and simplify xml
        (root,chars) = get_chars(root)
        phrases = chars2phrases(chars)

        # get axes, plot size and lines to delete
        if debug != 'get_axes':
            (axes, axes_min, axes_max, lines) = get_axes(lines,width,height)
        elif debug == 'get_axes':
            graphs = lines
            self.debug = {'lines':lines,
                          'width':width,
                          'height':height}

        # get graphs, areas and labels
        if (debug != 'get_graphs_areas'
        and debug != 'get_axes'):
            (graphs,areas,label_graphs) = get_graphs_areas(lines, axes, axes_min, axes_max, phrases)
        elif debug == 'get_graphs_areas':
            graphs = lines
            self.debug = {'lines':lines,
                          'axes':axes,
                          'axes_min':axes_min,
                          'axes_max':axes_max,
                          'phrases':phrases}

        #get bulletlines and labels
        bullet_lines, bullet_labels = get_bullets(curves,phrases)

        # connect graphs with the same style
        if (debug != 'get_graphs_areas'
        and debug != 'get_axes'
        and debug != 'connect_graphs'):
            graphs = connect_graphs(graphs, axes_min, axes_max)
        elif debug == 'connect_graphs':
            self.debug = {'graphs':graphs,
                          'axes_min':axes_min,
                          'axes_max':axes_max}

        # calibrate grid
        if (debug != 'calibrate_grid'
        and debug != 'get_graphs_areas'
        and debug != 'get_axes'
        and debug != 'connect_graphs'):
            grids = calibrate_grid(axes,phrases,width,height)
        elif debug == 'calibrate_grid':
            self.debug = {'axes':axes,
                          'phrases':phrases,
                          'width':width,
                          'height':height}

        # calibrate graphs
        if (debug != 'calibrate_graphs'
        and debug != 'calibrate_grid'
        and debug != 'get_graphs_areas'
        and debug != 'get_axes'
        and debug != 'connect_graphs'):
            graphs = calibrate_graphs(graphs,grids)
        elif debug ==  'calibrate_graphs':
            self.debug = {'axes':axes,
                          'phrases':phrases,
                          'width':width,
                          'height':height}

        # get labels for graphs
        if (debug != 'get_labels'
        and debug != 'calibrate_graphs'
        and debug != 'calibrate_grid'
        and debug != 'get_graphs_areas'
        and debug != 'get_axes'
        and debug != 'connect_graphs'):
            graphs = get_labels(graphs,label_graphs,bullet_lines,bullet_labels,grids,areas)
        elif debug == 'get_labels':
            self.debug = {'graphs':graphs,
                          'label_graphs':label_graphs,
                          'bullet_lines':bullet_lines,
                          'bullet_labels':bullet_labels,
                          'grids':grids,
                          'areas':areas}

        if (debug != 'calibrate_grid'
        and debug != 'get_graphs_areas'
        and debug != 'get_axes'
        and debug != 'connect_graphs'):
            self.grids = grids
        self.graphs = graphs

    def writesvg(self,filename):
        self._tree.write(filename)

    def plot(self):
        plot_graphs(self.graphs,self.grids)

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

def path_to_lines_and_curves(path_data): # SsQqTtAa is ignored
    path = parse_path(path_data)
    drawto_commands = 'MmZzLlHhVvCcSsQqTtAa'
    line_commands = 'LlHhVv'
    command = None
    lines = []
    line = []
    curves = []
    curve = []
    coordinate = []
    base_coordinate = [0,0]
    xy = 0
    for entry in path:
        if entry in drawto_commands:
            command  = entry
            absolute = entry.isupper()
            if command in 'Mm':
                if line:
                    lines.append(np.array(line))
                    line = []
                elif curve:
                    curves.append(np.array(curve))
                    curve = []
            elif command in 'Zz':
                if line and line[0] != line[-1]:
                    base_coordinate = line[0]
                    line.append(line[0])
                elif curve and curve[0] != curve[-1]:
                    base_coordinate = curve[0]
                    curve.append(curve[0])
            elif command in line_commands and curve:
                    curves.append(np.array(curve))
                    curve = []
            elif command in 'Cc' and line:
                if len(line)==1:
                    curve = line
                    line = []
                else:
                    lines.append(np.array(line))
                    line = []
        else:
            if not absolute:
                entry = base_coordinate[xy]+float(entry)
            else:
                entry = float(entry)
            if not xy and command not in 'HhVv':
                coordinate = [entry]
                xy = 1
            elif command in 'Hh':
                coordinate = [entry, base_coordinate[1]]
            elif command in 'Vv':
                coordinate = [base_coordinate[0], entry]
            else:
                coordinate.append(entry)
            if len(coordinate) == 2:
                xy = 0
                if (command in line_commands
                or command in 'Mm'):
                    base_coordinate = coordinate
                    line.append(coordinate)
                    if command in 'Mm':
                        if absolute:
                            command = 'L'
                        else:
                            command = 'l'
                elif command in 'Cc':
                    if len(curve) % 3 == 0:
                        base_coordinate = coordinate
                    curve.append(coordinate)
    if line:
        lines.append(np.array(line))
    if curve:
        curves.append(np.array(curve))
    return (lines,curves)

def lines_and_curves_to_path(lines,curves): # SsQqTtAa is ignored
    line_string = ''
    for line in lines:
        if line_string:
            line_string += ' '
        line_string += 'M'
        for coordinate in line.flatten():
            line_string += ' '+str(coordinate)
    for curve in curves:
        if line_string:
            line_string += ' '
        line_string += 'M'
        coordinates = curve.flatten()
        for i in range(len(coordinates)):
            coordinate = coordinates[i]
            if i == 2:
                line_string += ' C'
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
    chars_single = []
    for char in chars:
        if char in chars_single:
            next
        else:
            chars_single.append(char)
    chars_sorted = sorted(chars_single, key=itemgetter('x'), reverse=True)
    phrases = []
    while chars_sorted:
        startchar = chars_sorted.pop()
        phrase = startchar
        del_list = []
        superscript = 0
        subscript = 0
        ascent = 0
        descent = 0
        for i in range(len(chars_sorted)-1, -1, -1):
            testchar = chars_sorted[i]
            # testchar directly follows startchar
            if (testchar['y'] == phrase['y']
            and (testchar['x'] < startchar['x']+startchar['width']
                or (testchar['x'] < startchar['x']+startchar['width_space']*4.
                    and (testchar['text'] == '×' or startchar['text'] == '×'))
                )
            ):
                if superscript or subscript:
                    phrase['text'] += '}'
                    superscript = 0
                    subscript = 0
                phrase['text'] += testchar['text']
                startchar = testchar
                del_list.append(i)
            # superscript
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
                ascent_new = (phrase['y']-phrase['size'])-(testchar['y']-testchar['size'])
                if ascent_new > ascent:
                    ascent = ascent_new
                startchar = testchar
                del_list.append(i)
            # subscript
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
                descent_new = testchar['y']-phrase['y']
                if descent_new > descent:
                    descent = descent_new
                startchar = testchar
                del_list.append(i)
        if superscript or subscript:
            phrase['text'] += '}'
        length = startchar['x']+startchar['size']-phrase['x']
        phrase['dimensions'] = np.array([length,phrase['size']])
        phrase['asc_desc'] = np.array([ascent,descent])
        phrase['coords'] = np.array([phrase['x'],phrase['y']])
        del phrase['x']
        del phrase['y']
        del phrase['size']
        phrases.append(phrase)
        for i in del_list:
            del(chars_sorted[i])
    return phrases

def gridlines(axis, line_list, width, height, lines_to_delete): # edited
    gridlines = []
    if axis['min'][0]==axis['max'][0]:
        axis_type = 0
    elif axis['min'][1]==axis['max'][1]:
        axis_type = 1
    else: return [False, lines_to_delete]
    for line in line_list:
        length = line['max'][axis_type]-line['min'][axis_type]
        if (line['max'][axis_type]>=axis['min'][axis_type]-length/10
            and line['min'][axis_type]<=axis['min'][axis_type]+length/10
            and line['min'][1-axis_type]==line['max'][1-axis_type]
            and length < width/20
            and length < height/20
           ):
            gridline = {}
            gridline['length'] = length
            gridline['d'] = line['min']
            gridline['line'] = line
            gridlines.append(gridline)
            lines_to_delete.append(line)
    return [gridlines, lines_to_delete]

def get_lines_curves(root):
    lines = []
    curves = []
    paths = []
    for path in reversed(list(root.iter('{http://www.w3.org/2000/svg}path'))):
        if 'transform' in path.attrib:
            matrix = transform2matrix(path.attrib['transform'])
            line_list = []
            curve_list = []
            # transform path coordiantes
            lines_and_curves = path_to_lines_and_curves(path.attrib['d'])
            for line in lines_and_curves[0]:
                line_list.append(transform_path(line, matrix))
            for curve in lines_and_curves[1]:
                curve_list.append(transform_path(curve, matrix))
            # transform path style
            style = scale_style(path.attrib['style'], matrix)
            del path.attrib['transform']
        else:
            lines_and_curves = path_to_lines_and_curves(path.attrib['d'])
            (line_list,curve_list) = lines_and_curves
            if 'style' in path.attrib:
                style = path.attrib['style']
        path.attrib['style'] = style
        path.attrib['d'] = lines_and_curves_to_path(line_list,curve_list)
        if path.attrib not in paths:
            paths.append(path.attrib)
            # extract only information needed for data extraction
            for line in line_list:
                max_values = np.amax(line, axis=0)
                min_values = np.amin(line, axis=0)
                line_dict = {'d':line, 'style':style2dict(style), 'max':max_values, 'min':min_values}
                lines.append(line_dict)
            for curve in curve_list:
                max_values = np.amax(curve, axis=0)
                min_values = np.amin(curve, axis=0)
                curve_dict = {'d':curve,  'style':style2dict(style), 'max':max_values, 'min':min_values}
                curves.append(curve_dict)
        else:
            path.clear()
    return(root,lines,curves)

def get_chars(root):
    chars = []
    afm = {}
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
        else:
            matrix = np.identity(3)
        for tspan in text:
            (x,y)=transform_text(tspan.attrib['x'],tspan.attrib['y'],matrix)
            x_list = ast.literal_eval('['+x.replace(' ',',')+']')
            y_list = ast.literal_eval('['+y.replace(' ',',')+']')
            if 'style' in tspan.attrib:
                style = scale_style(tspan.attrib['style'], matrix)
            elif 'style' in text.attrib:
                style = scale_style(text.attrib['style'], matrix)
            style_dict = style2dict(style)
            text_attr = tspan.text
            if text_attr == None:
                text_attr = ''
            (size,unit) = font_size2size_units(style_dict['font-size'])
            if 'font-family' in style_dict:
                font = style_dict['font-family'].lower()
                if ('font-weight' in style_dict
                and style_dict['font-weight'].lower() == 'bold'):
                    font += ' '+'bold'
                if font not in afm:
                    afm[font] = get_font_metrics(font)
            else:
                font = None
            for i in range(len(text_attr)):
                char_text = text_attr[i]
                if font and afm[font]:
                    c = ord(char_text)
                    if c in afm[font]._metrics:
                        wx = afm[font]._metrics[c][0]
                        wx_space = afm[font]._metrics[32][0]
                        width = (wx+20)*(size)/1000.
                        width_space = (wx_space)*(size)/1000.
                    else:
                        width = size*0.891
                        width_space = size*.92
                else:
                    width = size*0.891
                    width_space = size*.92
                if len(x_list)==i:
                    last_width = chars[-1]['width']
                    x_list.append(x_list[i-1]+last_width*0.99)
                    y_list.append(y_list[i-1])
                x_val = x_list[i]
                y_val = y_list[i]
                char = {'text':char_text,'x':x_val,'y':y_val,
                        'size':size,'width':width,'width_space':width_space}
                chars.append(char)
            tspan.attrib['x'] = x
            tspan.attrib['y'] = y
            tspan.attrib['style'] = style
    return [root,chars]

def get_font_metrics(font):
    if font in afm_dict:
        afm_file = open(afm_dict[font], 'rb')
        afm = AFM(afm_file)
        afm_file.close()
    elif font in afm_translate:
        font_translated = afm_translate[font]
        afm_file = open(afm_dict[font_translated], 'rb')
        afm = AFM(afm_file)
        afm_file.close()
    else:
        afm = None
        #print(font+"\n")
    return afm

def get_axes(lines,width,height):
    axes = [[],[]]
    lines_to_delete = []
    for line in lines:
        if line['max'][0]-line['min'][0] > width/3:
            (grid, lines_to_delete) = gridlines(line, lines, width, height, lines_to_delete)
            if grid:
                line['grid']=grid
                if line['min'][1]/height < 0.5:
                    line['axis_pos'] = 0
                else:
                    line['axis_pos'] = 1
                axes[0].append(line)
                lines_to_delete.append(line)
        elif line['max'][1]-line['min'][1] > height/3:
            (grid, lines_to_delete) = gridlines(line, lines, width, height, lines_to_delete)
            if grid:
                line['grid']=grid
                if line['min'][0]/width > 0.5:
                    line['axis_pos'] = 0
                else:
                    line['axis_pos'] = 1
                axes[1].append(line)
                lines_to_delete.append(line)
    axes_positions = [[[],[]],[[],[]]]
    for i in range(2):
        for j in range(len(axes[i])):
            if axes[i][j]['axis_pos'] == 0:
                axes_positions[i][0].append(j)
            else:
                axes_positions[i][1].append(j)
    cleaned_axes = [[],[]]
    deleted_axes = []
    for i in range(2):
        axes_type = axes_positions[i]
        for axes_type_pos in axes_type:
            if len(axes_type_pos)>1:
                min_tot = np.array([width,height])
                max_tot = np.array([0.,0.])
                best_id = None
                num_grdlns = 0
                for j in axes_type_pos:
                    axis_min = axes[i][j]['min']
                    axis_num_grdlns = len(axes[i][j]['grid'])
                    if axis_min[1-i] < min_tot[1-i] and axis_num_grdlns >= num_grdlns:
                        min_tot = axis_min
                        num_grdlns = axis_num_grdlns
                        best_id = j
                for j in axes_type_pos:
                    if j == best_id:
                        for gridline in axes[i][j]['grid']:
                            del gridline['line']
                        cleaned_axes[i].append(axes[i][j])
                    else:
                        deleted_axes.append(axes[i][j])
            elif len(axes_type_pos)==1:
                j = axes_type_pos[0]
                for gridline in axes[i][j]['grid']:
                    del gridline['line']
                cleaned_axes[i].append(axes[i][j])
    axes = cleaned_axes
    axes_min = np.array([axes[0][0]['min'][0],axes[1][0]['min'][1]])
    axes_max = np.array([axes[0][0]['max'][0],axes[1][0]['max'][1]])
    new_lines = []
    for line in lines:
        save_line = False
        for deleted_axis in deleted_axes:
            for gridline in deleted_axis['grid']:
                if line is gridline['line']:
                    save_line = True
        if ((line['max']-.1 <= axes_max).all() and (line['min']+.1 >= axes_min).all()):
            delete = False
            for line_to_delete in lines_to_delete:
                if line is line_to_delete and not save_line:
                    delete = True
                    break
            if not delete:
                new_lines.append(line)
    lines = new_lines
    return [axes, axes_min, axes_max, lines]

def get_graphs_areas(lines, axes, axes_min, axes_max, phrases): # delete boundaries of areas with atol=0.02
    plot_size = axes_max-axes_min
    plot_width = plot_size[0]
    plot_height = plot_size[1]
    areas = []
    mesh = []
    graphs_prel = []
    graphs = []
    label_graphs = []
    for line in lines:
        if (
            (   line['style']['fill']
            and
                line['style']['fill'] != 'none')
        or
            (line['d'][0] == line['d'][-1]).all()
        ):
            areas.append(line)
        elif (
            line['max'][1] == line['min'][1]
        or
            line['max'][0] == line['min'][0]
        ):
            if (
                'stroke-dasharray' in line['style']
            and
                line['style']['stroke-dasharray']
            and
                line['style']['stroke-dasharray'] != 'none'
            and
                (   line['max'][0]-line['min'][0] > plot_width/2
                or
                    line['max'][1]-line['min'][1] > plot_height/2)
            ):
                mesh.append(line)
            else:
                graphs_prel.append(line)
        else:
            graphs_prel.append(line)
    for graph_prel in graphs_prel:
        boundary = 0
        if graph_prel['style']['stroke'] == 'none':
            boundary = 1
        elif graph_prel['min'][0] == graph_prel['max'][0]:
            boundary = 1 # vertical line is no graph
        elif graph_prel['max'][1] == graph_prel['min'][1]:
            for axis in axes[1]:
                if (axis['min'][1]+float(graph_prel['style']['stroke-width'])+2
                    > graph_prel['max'][1]):
                    boundary = 1
            arr1 = array_d_sort_x(graph_prel['d'])
            for area in areas:
                arr2 = array_d_sort_x(area['d'])
                if array_d_contained(arr1, arr2, atol=0.002):
                    boundary = 1
            graph_prel = get_line_label(graph_prel, phrases)
            if graph_prel['label_len'] > 0:
                label_graphs.append(graph_prel)
                boundary = 1
        if boundary == 0:
            graphs.append(graph_prel)
    return [graphs,areas,label_graphs]

def get_bullets(curves,phrases):
    bullet_lines = {}
    bullet_labels = {}
    for curve in curves:
        if (len(curve['d']) == 7
        and all(curve['d'][0] == curve['d'][-1])): # bullet
            bullet = get_line_label(curve, phrases)
            color = bullet['style']['fill']
            if bullet['label']:
                if color not in bullet_labels:
                    bullet_labels[color] = []
                bullet_labels[color].append(curve)
            else:
                if color not in bullet_lines:
                    bullet_lines[color] = []
                bullet_lines[color].append(curve)
    return (bullet_lines,bullet_labels)

def connect_graphs(graphs, axes_min, axes_max):
    to_delete = []
    plot_size = axes_max-axes_min
    graphs = sorted(graphs, key=lambda k: (k['min'][0]))
    connections = [{},{}]
    for i in range(len(graphs)):
        connections[0][i]={'id':i,'delta':float("inf")}
        connections[1][i]={'id':i,'delta':float("inf")}
        length_i = graphs[i]['d'][1]-graphs[i]['d'][0]
        slope_i = length_i[1]/length_i[0]
        for j in range(i+1, len(graphs)):
            length_j = graphs[j]['d'][1]-graphs[j]['d'][0]
            slope_j = length_j[1]/length_j[0]
            if (
                graphs[i]['style'] == graphs[j]['style']
            and
                len(graphs[i]['d']) == 2
            and
                len(graphs[j]['d']) == 2
            and
                abs(slope_i - slope_j) < 0.1
            and
                abs(slope_i) > 0.1
            ):
                to_delete.append(i)
                to_delete.append(j)
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            if (
                graphs[i]['style'] == graphs[j]['style']
            and
                graphs[i]['max'][0] <= graphs[j]['min'][0]+0.001
            and
                i not in to_delete
            and
                j not in to_delete
            ):
                delta = graphs[j]['min'][0] - graphs[i]['max'][0]
                if connections[0][i]['delta'] > delta:
                    connections[0][i]['delta'] = delta
                    connections[0][i]['id'] = j
                if connections[1][j]['delta'] > delta:
                    connections[1][j]['delta'] = delta
                    connections[1][j]['id'] = i

    for key, value in connections[0].items():
        if (
            value['id'] != key
        and
            connections[1][value['id']]['id'] == key
        ):
            graphs[value['id']]['d'] = np.concatenate((graphs[key]['d'], graphs[value['id']]['d']))
            graphs[value['id']]['max'][1] = max(graphs[value['id']]['max'][1], graphs[key]['max'][1])
            graphs[value['id']]['min'][0] = graphs[key]['min'][0]
            graphs[value['id']]['min'][1] = min(graphs[value['id']]['min'][1], graphs[key]['min'][1])
            graphs[value['id']]['connected'] = True
            to_delete.append(key)
    new_graphs = []
    for i in range(len(graphs)):
        graph_size = (graphs[i]['max'] - graphs[i]['min'])
        if ((graph_size > plot_size/9.5).any() and not i in to_delete):
            new_graphs.append(graphs[i])
    return new_graphs

def add_gridline_value(grid_calibr,gridline, phrases, axis, axis_type):
    for phrase in phrases:
        x = phrase['coords'][0]
        y = phrase['coords'][1]
        dx = phrase['dimensions'][0]
        dy = phrase['dimensions'][1]
        if ((y > gridline['d'][1]+gridline['length']
            and x < gridline['d'][0]
            and x+dx > gridline['d'][0]
            and y < axis['min'][1]+(dy+phrase['asc_desc'][0])*1.4
            and axis_type == 0)
        or
            (x+dx*0.7 < gridline['d'][0]
            and y-dy < gridline['d'][1]
            and y > gridline['d'][1]
            and x > gridline['d'][0]-dx*1.5
            and x > 0
            and axis_type == 1)
        ):
            gridline['text'] = phrase['text']
            text = gridline['text'];
            text = text.replace('−','-')
            if '^' in text:
                text = text.replace('}','')
                text = text.replace('0^{','E')
            float_strings = text.split("×")
            float_val = 1.
            for float_string in float_strings:
                float_val = float_val * float(float_string)
            gridline['value'] = float_val
            for gridtest in grid_calibr:
                if (gridtest['d'] == gridline['d']).all():
                    break
                elif (gridtest['value'] == gridline['value']
                and gridtest['length'] > gridline['length']):
                    break
            else:
                grid_calibr.append(gridline)
    return grid_calibr

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
                    if ((phrase['coords'][1] > axis['min'][1]+(phrase['dimensions'][1]+phrase['asc_desc'][0])*1.4 # x-axis label
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
                    if (grid_calibr and gridline['length'] < grid_calibr[-1]['length']):
                        break
                    else:
                        grid_calibr = add_gridline_value(grid_calibr,gridline, phrases, axis, axis_type)
                if (grid_calibr and len(grid_calibr) == 1):
                    grid = sorted(axis['grid'], key=lambda k: (k['d'][axis_type]))
                    for i in range(len(grid)):
                        if (i>0 and grid[i]['length']>grid[i-1]['length']):
                            v0 = grid[i]['value']
                            x0 = grid[i]['d'][axis_type]
                            v1 = v0*2
                            x1 = grid[i-1]['d'][axis_type]
                            break
                        elif (i<len(grid)-1 and grid[i]['length']>grid[i+1]['length']):
                            v0 = grid[i]['value']
                            x0 = grid[i]['d'][axis_type]
                            v1 = v0*0.9
                            x1 = grid[i+1]['d'][axis_type]
                            break
                    a = (x0-x1)/(np.log(v0)-np.log(v1))
                    b = (x1*np.log(v0)-x0*np.log(v1))/(np.log(v0)-np.log(v1))
                    x0 = grid[0]['d'][axis_type]
                    x1 = grid[1]['d'][axis_type]
                    x2 = grid[2]['d'][axis_type]
                    v0 = np.exp((x0-b)/a)
                    v1 = np.exp((x1-b)/a)
                    v2 = np.exp((x2-b)/a)
                    if (v0-v1)/(v1-v2)-1 < 0.001 :
                        for i in range(len(grid)):
                            x = grid[i]['d'][axis_type]
                            v = np.exp((x-b)/a)
                            grid[i]['value'] = v
                        grid_calibr = grid
                    else:
                        for gridline in grid_sorted:
                            grid_calibr = add_gridline_value(grid_calibr,gridline, phrases, axis, axis_type)
                        if (grid_calibr and len(grid_calibr) == 1):
                            raise Exception('Only one number on axis and no logscale found!')
                if (grid_calibr and len(grid_calibr) >= 3):
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
                elif (grid_calibr and len(grid_calibr) >= 2):
                    value0 = grid_calibr[0]['value']
                    value1 = grid_calibr[1]['value']
                    if (value0/value1 == 10 or value0/value1 == 1/10):
                        axis_scaling = 'log'
                        for gridline in grid_calibr:
                            gridline['value'] = np.log(gridline['value'])
                    else:
                        axis_scaling = 'linear'
                else:
                    raise Exception('no grid found!')
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

def array_d_sort_x(array):
    ind = np.argsort(array[:,0])
    return array[ind]

def array_d_contained(array1, array2, rtol=0, atol=0):
    i1 = 0
    i2 = 0
    match = 0
    if rtol == 0 and atol == 0:
        cmp_func = np.array_equal
    else:
        def cmp_func(arr1,arr2):
            return np.allclose(arr1,arr2,rtol,atol)
    while (i1 < len(array1) and i2 < len(array2)):
        if cmp_func(array1[i1],array2[i2]):
            i1 = i1+1
            i2 = i2+1
            match = match+1
        elif array1[i1][0] >= array2[i2][0] -atol-rtol*abs(array2[i2][0]):
            i2 = i2+1
        else:
            return False
        if match == len(array1):
            return True
    return False

def get_line_label(line, phrases):
    line['label'] = ''
    line['label_len'] = 0
    line['label_start'] = line['max'][0]
    for phrase in phrases:
        if line['label_start'] == line['max'][0]:
            label_start = line['label_start'] + phrase['dimensions'][1]*5
        else:
            label_start = line['label_start']
        if (phrase['coords'][0] > line['max'][0]
        and phrase['coords'][0]-phrase['dimensions'][1]*2 < label_start+line['label_len']*0.92
        and phrase['coords'][1]>line['min'][1]
        and phrase['coords'][1]-phrase['dimensions'][1]*0.7<line['max'][1]):
            line['label'] += ' '+phrase['text']
            if line['label_start'] == line['max'][0]:
                line['label_start'] = phrase['coords'][0]
            line['label_len'] += phrase['dimensions'][0]
    line['label']=line['label'].strip()
    return line

def get_labels(graphs,label_graphs,bullet_lines,bullet_labels,axes,areas):
    new_graphs = []
    gridline = axes[1]['grid'][0]
    label_min_x = gridline['d'][0] + gridline['length']
    assigned_label = []
    # add bullets to horizontal label_graphs with same style and different label
    for i in range(len(label_graphs)):
        for j in range(i):
            label_graph = label_graphs[i]
            cmp_label_graph = label_graphs[j]
            if (label_graph['min'][1] == label_graph['max'][1] # label_graph horizontal
            and cmp_label_graph['min'][1] == cmp_label_graph['max'][1] # cmp_label_graph horizontal
            and label_graph['style'] == cmp_label_graph['style'] # same style
            and label_graph['label'] != cmp_label_graph['label']): # different label
                for color, bullabs in bullet_labels.items():
                    for bullab in bullabs:
                        if bullab['label'] == label_graph['label']:
                            label_graph['style']['bullet'] = color
                            bullab['label_graph'] = True
                        elif bullab['label'] == cmp_label_graph['label']:
                            cmp_label_graph['style']['bullet'] = color
                            bullab['label_graph'] = True
    # add label to graphs from horizontal label_graphs with the same style
    for graph in graphs:
        graph['label'] = ''
        for i in range(len(label_graphs)):
            label_graph = label_graphs[i]
            if (label_graph['style'] == graph['style'] # same style
            and label_graph['min'][1] == label_graph['max'][1] # label_graph horizontal
            and not np.array_equal(label_graph['d'], graph['d']) # label_graph != graph
            and label_graph['min'][0]>label_min_x): # label_graph inside axes
                graph['label'] =  label_graph['label'].strip()
                graph['label_len'] = label_graph['label_len']
                assigned_label.append(i)
    # try to add not assigned labels to graphs without labels
    for graph in graphs:
        if graph['label'] =='':
            # try based on color and dashed / not dashed
            for i in range(len(label_graphs)):
                if i not in assigned_label:
                    label_graph = label_graphs[i]
                    if (label_graph['style']['stroke'] == graph['style']['stroke']
                    and ((label_graph['style']['stroke-dasharray'] != 'none'
                        and label_graph['style']['stroke-dasharray'] != ''
                        and graph['style']['stroke-dasharray'] != 'none'
                        and graph['style']['stroke-dasharray'] != '')
                        or (label_graph['style']['stroke-dasharray'] == 'none'
                            and graph['style']['stroke-dasharray'] == 'none'))
                    and label_graph['min'][1] == label_graph['max'][1]
                    and not np.array_equal(label_graph['d'], graph['d'])
                    and label_graph['min'][0]>label_min_x):
                        graph['label'] =  label_graph['label'].strip()
                        graph['label_len'] = label_graph['label_len']
                        assigned_label.append(i)
            # try based on bullets
            if graph['label'] =='':
                bullets = {}
                for color, bullines in bullet_lines.items():
                    for i in range(len(bullines)):
                        bulline = bullines[i]
                        bullet_point = (bulline['min']+bulline['max'])/2.
                        bullet_size = bulline['max']-bulline['min']
                        for coord in graph['d']:
                            if all(abs(coord-bullet_point) < bullet_size/1000):
                                if color not in bullets:
                                    bullets[color] = []
                                else:
                                    bullets[color].append(i)
                if bullets:
                    graph['bullets'] = bullets
                    for i in range(len(label_graphs)):
                        if (i not in assigned_label
                        and 'bullet' in label_graphs[i]['style']
                        and label_graphs[i]['style']['bullet'] in bullets):
                            graph['label'] =  label_graph['label'].strip()
                            graph['label_len'] = label_graph['label_len']
                            assigned_label.append(i)
                    for color, bullabs in bullet_labels.items():
                        for bullab in bullabs:
                            if ('label_graph' not in bullab
                            and color in bullets):
                                graph['label'] =  bullab['label'].strip()
                                graph['label_len'] = bullab['label_len']
    # compare graphs with eachother
    for i in range(len(graphs)):
        graph = graphs[i]
        graph['overlap'] = []
        graph['doubleline'] = False
        for j in range(len(graphs)):
            cmp_graph = graphs[j]
            # search for doublelines and overlaps
            if 'overlap' not in graph:
                graph['overlap'] = []
            if (((array_d_contained(cmp_graph['d'], graph['d'])
                    or array_d_contained(graph['d'], cmp_graph['d']))
                and cmp_graph['style'] != graph['style']
                and cmp_graph['label'] != ''
                and (graph['label']=='' or 'connected' in graph))
            or ('connected' in graph
                and 'connected' not in cmp_graph
                and cmp_graph['label'] == graph['label']
                and cmp_graph['label'] != '')
            ):
                graph['doubleline'] = True
                graph['overlap'].append(j)
            # use areas to assign different labels to
            # graphs with same style and label but different coordinates
            if (graph['label'] == cmp_graph['label']
            and graph['style'] == cmp_graph['style']
            and not np.array_equal(graph['d'], cmp_graph['d'])):
                for k in range(len(label_graphs)):
                    if (k not in assigned_label
                    and label_graph['min'][1] == label_graph['max'][1]
                    and label_graph['min'][0]>label_min_x):
                        for lg_g in (label_graph,graph):
                            if 'areas' not in lg_g:
                                lg_g['areas'] = []
                        for l in range(len(areas)):
                            area = areas[l]
                            for lg_g in (label_graph,graph):
                                if (area['min'][0] == lg_g['min'][0]
                                and area['max'][0] == lg_g['max'][0]
                                and area['min'][1] <= lg_g['min'][1]
                                and area['max'][1] >= lg_g['min'][1]
                                and l not in lg_g['areas']):
                                        lg_g['areas'].append(l)
                        for label_area in label_graph['areas']:
                            if 'areas' in graph:
                                for graph_area in graph['areas']:
                                    if (areas[label_area]['style']
                                    == areas[graph_area]['style']):
                                        graph['label'] =  label_graph['label']
                                        graph['label_len'] = label_graph['label_len']
    # change doubleline to other graph with overlap
    for graph in graphs:
        if ('overlap' in graph
        and graph['doubleline']
        and graph['label'] != ''):
            for overlap in graph['overlap']:
                for cmp_graph in graphs:
                    if (cmp_graph['label'] == graphs[overlap]['label']
                    and graphs[overlap]['label'] != graph['label']
                    and not np.array_equal(cmp_graph['d'], graphs[overlap]['d'])
                    and 'overlap' in cmp_graph
                    and overlap in cmp_graph['overlap']):
                        graph['doubleline'] = False
                        graphs[overlap]['doubleline'] = True
                        cmp_graph['doubleline'] = False
    # only add those graphs to new_graphs that are no doubleline
    for graph in graphs:
        if not graph['doubleline']:
            new_graphs.append(graph)
    graphs = new_graphs
    return graphs

def plot_graphs(graphs,grids_calibr):
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
            plt.setp(line, dashes=[2,2])
    #plt.title('title')
    if 'name' in grids_calibr[0]:
        xlabel = grids_calibr[0]['name'].replace(' ','\ ').replace('%','\%')
        plt.xlabel(r'$'+xlabel+'$')
    if 'name' in grids_calibr[1]:
        ylabel = grids_calibr[1]['name'].replace(' ','\ ').replace('%','\%')
        plt.ylabel(r'$'+ylabel+'$')
    plt.xscale(grids_calibr[0]['type'])
    plt.yscale(grids_calibr[1]['type'])
    plt.rcParams['legend.loc'] = 'best'
    plt.legend();
    plt.show()
