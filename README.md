Human sketch Recognition

Python code that recognizes human sketch objects. The original approach may be found here: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

Here we are using convolution neural network to recognize the sketch results:

There are several challenges (1) The number of labels are large : There are 255 different sketch object to recognize & many of them are vary similar(such as suv/van, chair/armchair bird/standing bird etc) (2) General publics are not very good artists. Many drawinings do not well reflect the intended object. (3) There are only 80 drawings per categories.

In the end, with limited data, large number of label, and confusing drawing , we are trying to get the best out of it.

dependencies

(1) PIL for reading and modifying png drawings

(2) Tensorflow and Keras for Convolution Neural Network libraryPIL 
