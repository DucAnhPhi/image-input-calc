# Image-Input Calculator

This program takes images of handwritten equations as an input and outputs the corresponding results.

## Installation

- install Python 3.6 (if not already installed)
- Set up a python virtual environment for this project. It keeps the dependencies/libraries required by different projects in separate places (isolates them to avoid conflicts).

```
$ pip install virtualenv
$ cd image-input-calc
$ virtualenv -p python3.6 virtual_env
```

- to begin using the virtual environment, it needs to be activated:

```
$ source virtual_env/bin/activate
```

- finally install all dependencies running:

```
$ pip install -r requirements.txt
```

- if you are done working in the virtual environment for the moment, you can deactivate it:

```
$ deactivate
```

## Application Flow

1. [Data Collection](#1.-data-collection)

2. [Pre-processing](#2.-pre-processing)

3. [Segmentation](#3.-segmentation)

4. [Ordering](#4.-ordering)

5. [Recognition](#5.-recognition)

6. [Symbolic Math Solver](#6.-symbolic-math-solver)

### 1. Data Collection
- use stream of images from webcam, containing handwritten equations

### 2. Pre-processing:
- add gaussian blur to image
- binary thresholding to make characters more distinct

### 3. Segmentation:
- automatically detect/crop each character from the frame:
-> detect contours with: satoshi suzuki et al. topological structural analysis of digitized binary images by border
-> approximations of polynomial representations of each contour are then generated(Ramer–Douglas–Peucker algorithm)
-> minimum sized bounding rectangles enclosing each contour are estimated

- after representing each character by bounding rectangle, characters with unrealistic dimensions (too big relative to others, too small, overlapping, etc.) can be discarded as noise.

- Each remaining character is then cropped from original frame and resized(typically shrunk) to a predetermined dimension so that ANN doesn't have to deal with huge varying-sized inputs

### 4. Ordering








- infer correct mathematical structure (ordering) of math symbols, letters and numbers
- apply local rules to comprehend compound fractions:

1. find widest division bar by evaluating each character's aspect ratio
2. look for characters whose centroids are above and below that division bar - group those characters into sub-expressions
3. start process over on each sub-expression
4. once all ratios evaluated, step backwards through recursive process and read left-to-right

### 5. Recognition
- [x] CNN to recognize single characters
- [ ] Training and test procedure for handwritten formulars
  - [ ] Direct segmented symbols as an input to the network
  - [ ] Outsource outliers
  - [ ] Training data?
  - [ ] Pretraining with MNIST?
 - [ ] Extract formula from recognized symbols
### 6. Symbolic math solver
- solve recognized equations using SymPy
