# svg2data
A Python module for reading data from a plot provided as SVG file.

- [Dependencies](#dependencies)
- [Usage](#usage)
  - [The svg2data class](#the-svg2data-class)
    - [Axes](#axes)
    - [Graphs](#graphs)
    - [Generating a simplified SVG file](#generating-a-simplified-svg-file)
    - [Plotting the extracted data](#plotting-the-extracted-data)
- [Converting a PDF to an SVG file](#converting-a-pdf-to-an-svg-file)

## Dependencies
In addition to the Python standard library, the **svg2data** module relies on the
packages **numpy** for numerical calculations and **pylab** for plotting.

## Usage
The main functionality is provided by the class `svg2data`.
It can be imported with

    import svg2data from svg2data

To use the plotting feature inside an IPython notebook, you may execute the
following line inside the notebook:

    %pylab inline

### The svg2data class
The class `svg2data` is used to read data from a SVG file containing a plot.
This is done by creating an instance of the `svg2data` class with the path to
the SVG file provided in the argument:

    data = svg2data("/path/to/file.svg")

#### Axes
All information on the axes of the plot are contained in a list of dictionaries
and can be accessed by

    axes = data.axes

#### Graphs
All information on graphs in the plot are contained in a list of dictionaries
and can be accessed by

    graphs = data.graphs

#### Generating a simplified SVG file
When processing the input SVG file, all contained coordinate transformations are
performed on the path elements (W3C provides detailed information on
[coordinate transformations](http://www.w3.org/TR/SVG/coords.html) and
[path elements](http://www.w3.org/TR/SVG/paths.html) in SVG files).
A new SVG file with all transformations on paths already performed (and some
further simplifications) can be written to disk using the `writesvg` method:

    data.writesvg('/path/to/newfile.svg')

This is in particular useful for verifying that all transformations were performed
properly.

#### Plotting the extracted data
To check if the extraction of the data from the original SVG file
was succesfull, the `plot` method can be used:

    data.plot()

This will produce a plot from the extracted data.

## Converting a PDF to an SVG file
Since many plots are only available as a PDF file, one has to convert them to SVG
for extracting data using **svg2data**. For this purpose, one can use
[**Inkscape**](http://inkscape.org/) from the command line:

    inkscape --without-gui --file=input.pdf --export-plain-svg=output.svg

SVG files created by [**pdf2svg**](https://github.com/db9052/pdf2svg) are **not
supported** since this tool converts text
to paths and thus prevents automatic extraction of the scales of the axes.
