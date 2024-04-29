# phase_based
PyTorch implementation of [Phase Based Motion Magnification](http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf). It is based off of MATLAB source that can be found [here](http://people.csail.mit.edu/nwadhwa/phase-video/), the input videos can also be found at this location. The PyTorch implementation allows for easily parallization on a GPU and is even much faster than a numpy implementation even without a GPU.

The [main notebook](https://github.com/itberrios/phase_based/blob/main/motion_amplification_pytorch.ipynb) contains a detailed hands-on overview of the Motion Magnification Algorithm. The image below shows an example of how the motion is amplified across all the video frames.
![image](https://github.com/itberrios/phase_based/assets/60835780/7dd5b6b0-1e6e-423b-95d0-4dedbce7065f)


<br>

While the notebook is good for interactive use, the main script called motion_magnification.py is a better option to run on your own videos. It can be called from the commandline or alternatively the arguments can be input directly in the script. 

## Applying Motion Magnification

The following commandline arguments produce the following GIF: <br>
``` python motion_magnification.py -v videos/crane_crop.avi -a 25 -lo 0.2 -hi 0.25 -n luma3 -p half_octave -s 5.0 -b 4 -c 0.7 -gif True ``` 

![crane_crop_luma3_25x](https://github.com/itberrios/phase_based/assets/60835780/83cebe8d-eafa-4342-b5c1-2a9cc13ea458)

### Arguments:
A list of the arguments is provided below. Please use the help option to find more info: 
``` python motion_magnification.py --help ```

- --video_path, -v         **&rarr;** Path to input video (**Required**)
- --phase_mag, -a          **&rarr;** Phase Magnification Factor (**Required**)
- --freq_lo, -lo           **&rarr;** Low Frequency cutoff for Temporal Filter (**Required**)
- --freq_hi, -hi           **&rarr;** High Frequency cutoff for Temporal Filter (**Required**)
- --colorspace, -n         **&rarr;** Colorspace for processing
- --pyramid_type, -p       **&rarr;** Complex Steerable Pyramid Type
- --sigma, -s              **&rarr;** Gaussian Kernel for Phase Filtering
- --attenuate, -a          **&rarr;** Attenuates Other frequencies outside of lo and hi
- --sample_frequency, -fs  **&rarr;** Overrides video sample frequency
- --reference_index, -r    **&rarr;** Index of DC reference frame
- --scale_factor, -c       **&rarr;** Factor to scale frames for processing
- --batch_size, -b         **&rarr;** CUDA batch size
- --save_directory, -d     **&rarr;** Directory for output files (default is input video directory)  
- --save_gif, -gif         **&rarr;** Saves results as a GIF
