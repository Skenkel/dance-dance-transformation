# dance-dance-transformation
DDR simfile generator using a transformer architecture


# Dummy Dataset
Get data from the drive. 

/data/audio has just the audio files
/chart_pkl has the ddc style charts within the included chart class.
/ddc_style_feats has the features extracted by ddc and pickled
/json and /json_filtered have the simnfiles converted to json
original has the simfile, audio etc(for use with stepmania)





Scratchpad2.py attempts to load the data (some text file hacking will be needed)

How it works:
there is a .pkl which contains, as one object, the numpy arrays for extracted audio features, as well as the step files for different difficulties. The unit of time is a "frame". 

Next step:
Edit encoder/Decoder to parse DDC frame arrays. 




some of the code is adapted from https://github.com/chrisdonahue/ddc
some is inspired by https://github.com/stonyhu/DanceRevolution
